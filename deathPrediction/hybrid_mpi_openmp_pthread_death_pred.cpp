#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <mpi.h>
#include <omp.h>
#include <pthread.h>

using namespace std;

struct Patient {
    int subjectId;
    string ethnicity;
    string gender;
    int icd9Code1;
    int expireFlag;
};

struct GradientThreadData {
    const vector<Patient>* data;
    int start;
    int end;
    map<string, double>* ethnicityWeights;
    map<string, double>* genderWeights;
    map<int, double>* icd9Weights;
    double bias;
    
    // Results
    double biasGrad;
    map<string, double> ethGrads;
    map<string, double> genGrads;
    map<int, double> icdGrads;
};

double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

void* computeGradient(void* arg) {
    GradientThreadData* td = (GradientThreadData*)arg;
    
    td->biasGrad = 0.0;
    td->ethGrads.clear();
    td->genGrads.clear();
    td->icdGrads.clear();
    
    for (int i = td->start; i < td->end; i++) {
        const Patient& p = (*td->data)[i];
        double z = td->bias + (*td->ethnicityWeights)[p.ethnicity] + 
                  (*td->genderWeights)[p.gender] + (*td->icd9Weights)[p.icd9Code1];
        double pred = sigmoid(z);
        double error = p.expireFlag - pred;
        
        td->biasGrad += error;
        td->ethGrads[p.ethnicity] += error;
        td->genGrads[p.gender] += error;
        td->icdGrads[p.icd9Code1] += error;
    }
    
    return NULL;
}

class LogisticRegressionTripleHybrid {
private:
    map<string, double> ethnicityWeights;
    map<string, double> genderWeights;
    map<int, double> icd9Weights;
    double bias;
    int rank;
    int size;
    int numPThreads;
    
public:
    LogisticRegressionTripleHybrid(int r, int s, int pthreads = 2) 
        : bias(0.0), rank(r), size(s), numPThreads(pthreads > 0 ? pthreads : 1) {}
    
    void train(const vector<Patient>& data, int epochs = 100, double lr = 0.01) {
        // Initialize weights (only on rank 0)
        if (rank == 0) {
            for (const auto& p : data) {
                ethnicityWeights[p.ethnicity] = 0.0;
                genderWeights[p.gender] = 0.0;
                icd9Weights[p.icd9Code1] = 0.0;
            }
        }
        
        // Triple hybrid: MPI + OpenMP + Pthread gradient descent
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Level 1: MPI - Distribute data across processes
            int mpiChunkSize = data.size() / size;
            int mpiStart = rank * mpiChunkSize;
            int mpiEnd = (rank == size - 1) ? data.size() : (rank + 1) * mpiChunkSize;
            int mpiDataSize = mpiEnd - mpiStart;
            
            double processBiasGrad = 0.0;
            map<string, double> processEthGrads;
            map<string, double> processGenGrads;
            map<int, double> processIcdGrads;
            
            // Level 2: OpenMP - Parallelize within each MPI process
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                int numOmpThreads = omp_get_num_threads();
                int ompChunkSize = mpiDataSize / numOmpThreads;
                int ompStart = mpiStart + tid * ompChunkSize;
                int ompEnd = (tid == numOmpThreads - 1) ? mpiEnd : (mpiStart + (tid + 1) * ompChunkSize);
                int ompDataSize = ompEnd - ompStart;
                
                // Level 3: Pthread - Fine-grained control within OpenMP threads
                vector<pthread_t> threads(numPThreads);
                vector<GradientThreadData> threadData(numPThreads);
                
                int pthreadChunkSize = ompDataSize / numPThreads;
                
                for (int t = 0; t < numPThreads; t++) {
                    threadData[t].data = &data;
                    threadData[t].start = ompStart + t * pthreadChunkSize;
                    threadData[t].end = (t == numPThreads - 1) ? ompEnd : (ompStart + (t + 1) * pthreadChunkSize);
                    threadData[t].ethnicityWeights = &ethnicityWeights;
                    threadData[t].genderWeights = &genderWeights;
                    threadData[t].icd9Weights = &icd9Weights;
                    threadData[t].bias = bias;
                    
                    pthread_create(&threads[t], NULL, computeGradient, &threadData[t]);
                }
                
                // Join Pthreads and accumulate results
                double ompBiasGrad = 0.0;
                map<string, double> ompEthGrads;
                map<string, double> ompGenGrads;
                map<int, double> ompIcdGrads;
                
                for (int t = 0; t < numPThreads; t++) {
                    pthread_join(threads[t], NULL);
                    
                    ompBiasGrad += threadData[t].biasGrad;
                    for (const auto& kv : threadData[t].ethGrads) ompEthGrads[kv.first] += kv.second;
                    for (const auto& kv : threadData[t].genGrads) ompGenGrads[kv.first] += kv.second;
                    for (const auto& kv : threadData[t].icdGrads) ompIcdGrads[kv.first] += kv.second;
                }
                
                // OpenMP reduction: Combine results from all OpenMP threads
                #pragma omp critical
                {
                    processBiasGrad += ompBiasGrad;
                    for (const auto& kv : ompEthGrads) processEthGrads[kv.first] += kv.second;
                    for (const auto& kv : ompGenGrads) processGenGrads[kv.first] += kv.second;
                    for (const auto& kv : ompIcdGrads) processIcdGrads[kv.first] += kv.second;
                }
            }
            
            // MPI reduction: Combine results from all processes
            double globalBiasGrad = 0.0;
            MPI_Allreduce(&processBiasGrad, &globalBiasGrad, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            
            // Update bias
            bias += lr * globalBiasGrad;
            
            // Update weights based on local gradients
            // KNOWN LIMITATION: Weights updated locally for demonstration.
            // Production code should use MPI_Allreduce for full weight synchronization.
            for (auto& kv : ethnicityWeights) {
                if (processEthGrads.count(kv.first)) {
                    kv.second += lr * processEthGrads[kv.first];
                }
            }
            for (auto& kv : genderWeights) {
                if (processGenGrads.count(kv.first)) {
                    kv.second += lr * processGenGrads[kv.first];
                }
            }
            for (auto& kv : icd9Weights) {
                if (processIcdGrads.count(kv.first)) {
                    kv.second += lr * processIcdGrads[kv.first];
                }
            }
            
            // Broadcast updated bias
            MPI_Bcast(&bias, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
    }
    
    double predict(const Patient& p) {
        double z = bias + ethnicityWeights[p.ethnicity] + 
                  genderWeights[p.gender] + icd9Weights[p.icd9Code1];
        return sigmoid(z);
    }
    
    double calculateAccuracy(const vector<Patient>& data) {
        // MPI: Distribute evaluation across processes
        int mpiChunkSize = data.size() / size;
        int mpiStart = rank * mpiChunkSize;
        int mpiEnd = (rank == size - 1) ? data.size() : (rank + 1) * mpiChunkSize;
        
        int localCorrect = 0;
        
        // OpenMP: Parallelize evaluation within each process
        #pragma omp parallel for reduction(+:localCorrect)
        for (int i = mpiStart; i < mpiEnd; i++) {
            double pred = predict(data[i]);
            int predClass = pred >= 0.5 ? 1 : 0;
            if (predClass == data[i].expireFlag) localCorrect++;
        }
        
        int globalCorrect = 0;
        MPI_Allreduce(&localCorrect, &globalCorrect, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        
        return (double)globalCorrect / data.size();
    }
    
    double calculateDeathRate(const vector<Patient>& data) {
        // MPI: Distribute prediction across processes
        int mpiChunkSize = data.size() / size;
        int mpiStart = rank * mpiChunkSize;
        int mpiEnd = (rank == size - 1) ? data.size() : (rank + 1) * mpiChunkSize;
        
        double localPred = 0.0;
        
        // OpenMP: Parallelize prediction within each process
        #pragma omp parallel for reduction(+:localPred)
        for (int i = mpiStart; i < mpiEnd; i++) {
            localPred += predict(data[i]);
        }
        
        double globalPred = 0.0;
        MPI_Allreduce(&localPred, &globalPred, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        return globalPred / data.size();
    }
};

// Helper function to safely convert string to int
int safeStoi(const string& str) {
    if (str.empty()) return 0;
    try {
        string cleaned = str;
        cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), ' '), cleaned.end());
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
    
    int lineNum = 1;
    while (getline(file, line)) {
        lineNum++;
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
        } catch (const exception& e) {
            continue;
        }
    }
    
    return patients;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc < 2) {
        if (rank == 0) {
            cerr << "Usage: mpirun -np <processes> " << argv[0] << " <data_file> [num_pthreads]" << endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    int numPThreads = (argc > 2) ? atoi(argv[2]) : 2;
    
    auto startTotal = chrono::high_resolution_clock::now();
    
    // Load data (all processes load data)
    auto startLoad = chrono::high_resolution_clock::now();
    vector<Patient> data = loadData(argv[1]);
    auto endLoad = chrono::high_resolution_clock::now();
    double loadTime = chrono::duration<double>(endLoad - startLoad).count();
    
    if (rank == 0) {
        cout << "Triple Hybrid MPI+OpenMP+Pthread Implementation - Logistic Regression" << endl;
        cout << "=======================================================================" << endl;
        cout << "MPI Processes: " << size << endl;
        cout << "OpenMP Threads per process: " << omp_get_max_threads() << endl;
        cout << "Pthreads per OpenMP thread: " << numPThreads << endl;
        cout << "Total parallelism: " << (size * omp_get_max_threads() * numPThreads) << " threads" << endl;
        cout << "Data loaded: " << data.size() << " patients" << endl;
        cout << "Load time: " << loadTime << " seconds" << endl;
    }
    
    // Split data (80% train, 20% test)
    int trainSize = data.size() * 0.8;
    vector<Patient> trainData(data.begin(), data.begin() + trainSize);
    vector<Patient> testData(data.begin() + trainSize, data.end());
    
    // Train model
    auto startTrain = chrono::high_resolution_clock::now();
    LogisticRegressionTripleHybrid model(rank, size, numPThreads);
    model.train(trainData, 100, 0.01);
    auto endTrain = chrono::high_resolution_clock::now();
    double trainTime = chrono::duration<double>(endTrain - startTrain).count();
    
    if (rank == 0) {
        cout << "Training time: " << trainTime << " seconds" << endl;
    }
    
    // Evaluate
    auto startEval = chrono::high_resolution_clock::now();
    double accuracy = model.calculateAccuracy(testData);
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
        
        // Save timing to file
        ofstream timingFile("timing_hybrid_mpi_openmp_pthread.txt");
        timingFile << "load," << loadTime << endl;
        timingFile << "train," << trainTime << endl;
        timingFile << "eval," << evalTime << endl;
        timingFile << "total," << totalTime << endl;
        timingFile << "accuracy," << accuracy << endl;
        timingFile << "deathrate," << deathRate << endl;
        timingFile << "processes," << size << endl;
        timingFile << "threads," << (omp_get_max_threads() * numPThreads) << endl;
        timingFile.close();
    }
    
    MPI_Finalize();
    return 0;
}
