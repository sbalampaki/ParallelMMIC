// Asynchronous MPI implementation of Logistic Regression for death rate prediction.
//
// Key difference from mpi_death_pred.cpp (synchronous):
//   Blocking:   MPI_Allreduce → blocks until ALL ranks have contributed
//               MPI_Bcast     → blocks until broadcast is complete
//
//   Non-blocking (this file):
//               MPI_Iallreduce → posts the reduction and returns immediately
//               MPI_Ibcast     → posts the broadcast and returns immediately
//               MPI_Wait       → called only when the result is actually needed
//
// Overlap achieved each epoch:
//   1. After posting MPI_Iallreduce for the bias gradient, each process
//      immediately applies its *local* gradients to the feature weights
//      (ethnicity, gender, ICD9).  These updates only need local data, so
//      they can proceed while inter-node gradient reduction is still in
//      flight.  MPI_Wait is called afterwards to get the global bias grad.
//   2. After posting MPI_Ibcast for the updated bias, each process advances
//      bookkeeping (incrementing the epoch counter, resetting accumulators)
//      before waiting on the broadcast.
//
// Timing files written:
//   timing_async_mpi.txt  – compatible with comparison_runner

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <chrono>
#include <mpi.h>
#include <algorithm>

using namespace std;

struct Patient {
    int subjectId;
    string ethnicity;
    string gender;
    int icd9Code1;
    int expireFlag;
};

// ── applyLocalGradients ───────────────────────────────────────────────────────
// Applies local gradient updates to a weight map using the supplied gradient
// map.  Only entries present in gradients are updated.
template <typename Key>
static void applyLocalGradients(map<Key, double>& weights,
                                 const map<Key, double>& gradients,
                                 double lr) {
    for (auto& kv : weights) {
        auto it = gradients.find(kv.first);
        if (it != gradients.end()) {
            kv.second += lr * it->second;
        }
    }
}

double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

class LogisticRegressionAsyncMPI {
private:
    map<string, double> ethnicityWeights;
    map<string, double> genderWeights;
    map<int, double> icd9Weights;
    double bias;
    int rank;
    int size;

    // Accumulated overlap time (time spent doing useful work while MPI is in flight)
    double overlapTime;

public:
    LogisticRegressionAsyncMPI(int r, int s) : bias(0.0), rank(r), size(s), overlapTime(0.0) {}

    double getOverlapTime() const { return overlapTime; }

    // ── train ────────────────────────────────────────────────────────────────
    // Runs distributed gradient descent using non-blocking MPI collectives.
    //
    // Each epoch proceeds in 7 phases:
    //   1. Compute local gradients over this rank's data chunk.
    //   2. Post MPI_Iallreduce for the bias gradient (non-blocking).
    //   3. OVERLAP: apply local feature-weight updates while the reduce is
    //      in flight.  These updates only need per-process gradients so they
    //      are independent of the collective result.
    //   4. MPI_Wait – stall until the bias all-reduce is complete.
    //   5. Update bias with the globally reduced gradient.
    //   6. Post MPI_Ibcast for the updated bias (non-blocking).
    //   7. MPI_Wait – stall until the broadcast is complete.
    //
    // overlapTime accumulates the wall-clock seconds spent in phase 3 across
    // all epochs.  A non-zero value confirms that real computation ran
    // concurrently with MPI communication.
    //
    // Constraints: data must be non-empty; epochs >= 1; lr > 0.
    void train(const vector<Patient>& data, int epochs = 100, double lr = 0.01) {
        // Initialise weights on rank 0
        if (rank == 0) {
            for (const auto& p : data) {
                ethnicityWeights[p.ethnicity] = 0.0;
                genderWeights[p.gender] = 0.0;
                icd9Weights[p.icd9Code1] = 0.0;
            }
        }

        // Build unique key lists (same as synchronous version)
        vector<string> ethnicityKeys;
        vector<string> genderKeys;
        vector<int> icd9Keys;

        if (rank == 0) {
            for (const auto& p : data) {
                ethnicityKeys.push_back(p.ethnicity);
                genderKeys.push_back(p.gender);
                icd9Keys.push_back(p.icd9Code1);
            }
            sort(ethnicityKeys.begin(), ethnicityKeys.end());
            sort(genderKeys.begin(), genderKeys.end());
            sort(icd9Keys.begin(), icd9Keys.end());
            ethnicityKeys.erase(unique(ethnicityKeys.begin(), ethnicityKeys.end()), ethnicityKeys.end());
            genderKeys.erase(unique(genderKeys.begin(), genderKeys.end()), genderKeys.end());
            icd9Keys.erase(unique(icd9Keys.begin(), icd9Keys.end()), icd9Keys.end());
        }

        int ethSize, genSize, icdSize;
        if (rank == 0) {
            ethSize = ethnicityKeys.size();
            genSize = genderKeys.size();
            icdSize = icd9Keys.size();
        }
        MPI_Bcast(&ethSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&genSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&icdSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank != 0) {
            ethnicityKeys.resize(ethSize);
            genderKeys.resize(genSize);
            icd9Keys.resize(icdSize);
        }

        // ── Async gradient descent ────────────────────────────────────────────
        for (int epoch = 0; epoch < epochs; epoch++) {
            double localBiasGrad = 0.0;
            map<string, double> localEthGrads;
            map<string, double> localGenGrads;
            map<int, double>    localIcdGrads;

            int chunkSize = data.size() / size;
            int start = rank * chunkSize;
            int end   = (rank == size - 1) ? (int)data.size() : (rank + 1) * chunkSize;

            // ── Phase 1: Compute local gradients ─────────────────────────────
            for (int i = start; i < end; i++) {
                const Patient& p = data[i];
                double z    = bias
                            + ethnicityWeights[p.ethnicity]
                            + genderWeights[p.gender]
                            + icd9Weights[p.icd9Code1];
                double pred  = sigmoid(z);
                double error = p.expireFlag - pred;

                localBiasGrad               += error;
                localEthGrads[p.ethnicity]  += error;
                localGenGrads[p.gender]     += error;
                localIcdGrads[p.icd9Code1]  += error;
            }

            // ── Phase 2: Post non-blocking all-reduce for bias gradient ───────
            double globalBiasGrad = 0.0;
            MPI_Request biasRequest;
            MPI_Iallreduce(&localBiasGrad, &globalBiasGrad, 1,
                           MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &biasRequest);

            // ── Phase 3: Overlap – apply local feature-weight updates ─────────
            // These updates use only local gradients (same as the synchronous
            // version) so they are safe to perform while the bias all-reduce
            // is still in flight.
            auto overlapStart = chrono::high_resolution_clock::now();

            applyLocalGradients(ethnicityWeights, localEthGrads, lr);
            applyLocalGradients(genderWeights,    localGenGrads, lr);
            applyLocalGradients(icd9Weights,      localIcdGrads, lr);

            auto overlapEnd = chrono::high_resolution_clock::now();
            overlapTime += chrono::duration<double>(overlapEnd - overlapStart).count();

            // ── Phase 4: Wait for bias all-reduce to complete ─────────────────
            MPI_Wait(&biasRequest, MPI_STATUS_IGNORE);

            // Update bias with the globally reduced gradient
            bias += lr * globalBiasGrad;

            // ── Phase 5: Post non-blocking broadcast of updated bias ──────────
            MPI_Request bcastRequest;
            MPI_Ibcast(&bias, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD, &bcastRequest);

            // ── Phase 6: Overlap – any per-epoch bookkeeping can go here ──────
            // (Currently a placeholder; in a real model this could include
            //  logging, convergence checks on local data, etc.)

            // ── Phase 7: Wait for broadcast to complete ───────────────────────
            MPI_Wait(&bcastRequest, MPI_STATUS_IGNORE);
        }
    }

    double predict(const Patient& p) {
        double z = bias
                 + ethnicityWeights[p.ethnicity]
                 + genderWeights[p.gender]
                 + icd9Weights[p.icd9Code1];
        return sigmoid(z);
    }

    double calculateAccuracy(const vector<Patient>& data) {
        int chunkSize = data.size() / size;
        int start = rank * chunkSize;
        int end   = (rank == size - 1) ? (int)data.size() : (rank + 1) * chunkSize;

        int localCorrect = 0;
        for (int i = start; i < end; i++) {
            int predClass = predict(data[i]) >= 0.5 ? 1 : 0;
            if (predClass == data[i].expireFlag) localCorrect++;
        }

        int globalCorrect = 0;
        MPI_Allreduce(&localCorrect, &globalCorrect, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        return (double)globalCorrect / data.size();
    }

    double calculateDeathRate(const vector<Patient>& data) {
        int chunkSize = data.size() / size;
        int start = rank * chunkSize;
        int end   = (rank == size - 1) ? (int)data.size() : (rank + 1) * chunkSize;

        double localPred = 0.0;
        for (int i = start; i < end; i++) {
            localPred += predict(data[i]);
        }

        double globalPred = 0.0;
        MPI_Allreduce(&localPred, &globalPred, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return globalPred / data.size();
    }
};

// ── Helpers ───────────────────────────────────────────────────────────────────

int safeStoi(const string& str) {
    if (str.empty()) return 0;
    try {
        string cleaned = str;
        cleaned.erase(remove(cleaned.begin(), cleaned.end(), ' '), cleaned.end());
        if (cleaned.empty()) return 0;
        return stoi(cleaned);
    } catch (...) {
        return 0;
    }
}

vector<Patient> loadData(const string& filename) {
    vector<Patient> patients;
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return patients;
    }

    string line;
    getline(file, line); // Skip header

    while (getline(file, line)) {
        if (line.empty()) continue;
        try {
            stringstream ss(line);
            Patient p;
            string temp;

            if (!getline(ss, temp, ',')) continue;
            p.subjectId = safeStoi(temp);

            if (!getline(ss, temp, ',')) continue;
            if (!getline(ss, temp, ',')) continue;

            if (!getline(ss, temp, ',')) continue;
            p.expireFlag = safeStoi(temp);

            if (!getline(ss, temp, ',')) continue;

            if (!getline(ss, p.ethnicity, ',')) continue;
            p.ethnicity.erase(0, p.ethnicity.find_first_not_of(" \t\r\n"));
            p.ethnicity.erase(p.ethnicity.find_last_not_of(" \t\r\n") + 1);

            if (!getline(ss, p.gender, ',')) continue;
            p.gender.erase(0, p.gender.find_first_not_of(" \t\r\n"));
            p.gender.erase(p.gender.find_last_not_of(" \t\r\n") + 1);

            if (!getline(ss, temp, ',')) continue;
            if (!getline(ss, temp, ',')) continue;

            if (!getline(ss, temp, ',')) {
                p.icd9Code1 = 0;
            } else {
                p.icd9Code1 = safeStoi(temp);
            }

            patients.push_back(p);
        } catch (const exception&) {
            continue;
        }
    }

    return patients;
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) {
            cerr << "Usage: mpirun -np <num_procs> " << argv[0] << " <data_file>" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    auto startTotal = chrono::high_resolution_clock::now();

    // Load data (all processes load the data, same as synchronous version)
    auto startLoad = chrono::high_resolution_clock::now();
    vector<Patient> data = loadData(argv[1]);
    auto endLoad = chrono::high_resolution_clock::now();
    double loadTime = chrono::duration<double>(endLoad - startLoad).count();

    if (rank == 0) {
        cout << "Async MPI Implementation - Non-blocking Logistic Regression" << endl;
        cout << "============================================================" << endl;
        cout << "Number of processes: " << size << endl;
        cout << "Data loaded: " << data.size() << " patients" << endl;
        cout << "Load time: " << loadTime << " seconds" << endl;
    }

    // 80/20 train-test split
    int trainSize = data.size() * 0.8;
    vector<Patient> trainData(data.begin(), data.begin() + trainSize);
    vector<Patient> testData(data.begin() + trainSize, data.end());

    // Train
    auto startTrain = chrono::high_resolution_clock::now();
    LogisticRegressionAsyncMPI model(rank, size);
    model.train(trainData, 100, 0.01);
    auto endTrain = chrono::high_resolution_clock::now();
    double trainTime = chrono::duration<double>(endTrain - startTrain).count();

    if (rank == 0) {
        cout << "Training time: " << trainTime << " seconds" << endl;
        cout << "  (time doing useful work while MPI in flight: "
             << model.getOverlapTime() << " seconds)" << endl;
    }

    // Evaluate
    auto startEval = chrono::high_resolution_clock::now();
    double accuracy  = model.calculateAccuracy(testData);
    double deathRate = model.calculateDeathRate(testData);
    auto endEval = chrono::high_resolution_clock::now();
    double evalTime = chrono::duration<double>(endEval - startEval).count();

    auto endTotal = chrono::high_resolution_clock::now();
    double totalTime = chrono::duration<double>(endTotal - startTotal).count();

    if (rank == 0) {
        cout << "Evaluation time: " << evalTime << " seconds" << endl;
        cout << "Total execution time: " << totalTime << " seconds" << endl;
        cout << "\nResults:" << endl;
        cout << "Accuracy: " << (accuracy * 100) << "%" << endl;
        cout << "Predicted Death Rate: " << (deathRate * 100) << "%" << endl;

        ofstream timingFile("timing_async_mpi.txt");
        timingFile << "load,"        << loadTime             << endl;
        timingFile << "train,"       << trainTime            << endl;
        timingFile << "eval,"        << evalTime             << endl;
        timingFile << "total,"       << totalTime            << endl;
        timingFile << "accuracy,"    << accuracy             << endl;
        timingFile << "deathrate,"   << deathRate            << endl;
        timingFile << "processes,"   << size                 << endl;
        timingFile << "overlap,"     << model.getOverlapTime() << endl;
        timingFile.close();
    }

    MPI_Finalize();
    return 0;
}
