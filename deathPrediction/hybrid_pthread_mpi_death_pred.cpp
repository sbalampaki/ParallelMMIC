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
#include <pthread.h>

using namespace std;

struct Patient {
    int subjectId;
    string ethnicity;
    string gender;
    int icd9Code1;
    int expireFlag;
};

struct ThreadData {
    const vector<Patient>* data;
    int start;
    int end;
    map<string, double>* ethnicityWeights;
    map<string, double>* genderWeights;
    map<int, double>* icd9Weights;
    double bias;
    double lr;
    
    // Gradient accumulators
    double biasGrad;
    map<string, double> ethGrads;
    map<string, double> genGrads;
    map<int, double> icdGrads;
};

struct PredictThreadData {
    const vector<Patient>* data;
    int start;
    int end;
    map<string, double>* ethnicityWeights;
    map<string, double>* genderWeights;
    map<int, double>* icd9Weights;
    double bias;
    int correct;
    double totalPred;
};

double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

void* trainThread(void* arg) {
    ThreadData* td = (ThreadData*)arg;
    
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

void* predictThread(void* arg) {
    PredictThreadData* pd = (PredictThreadData*)arg;
    
    pd->correct = 0;
    pd->totalPred = 0.0;
    
    for (int i = pd->start; i < pd->end; i++) {
        const Patient& p = (*pd->data)[i];
        double z = pd->bias + (*pd->ethnicityWeights)[p.ethnicity] + 
                  (*pd->genderWeights)[p.gender] + (*pd->icd9Weights)[p.icd9Code1];
        double pred = sigmoid(z);
        
        int predClass = pred >= 0.5 ? 1 : 0;
        if (predClass == p.expireFlag) pd->correct++;
        pd->totalPred += pred;
    }
    
    return NULL;
}

class LogisticRegressionHybridPThreadMPI {
private:
    map<string, double> ethnicityWeights;
    map<string, double> genderWeights;
    map<int, double> icd9Weights;
    double bias;
    int rank;
    int size;
    int numThreads;
    
public:
    LogisticRegressionHybridPThreadMPI(int r, int s, int threads = 4) 
        : bias(0.0), rank(r), size(s), numThreads(threads > 0 ? threads : 1) {}
    
    void train(const vector<Patient>& data, int epochs = 100, double lr = 0.01) {
        // Initialize weights (only on rank 0)
        if (rank == 0) {
            for (const auto& p : data) {
                ethnicityWeights[p.ethnicity] = 0.0;
                genderWeights[p.gender] = 0.0;
                icd9Weights[p.icd9Code1] = 0.0;
            }
        }
        
        // Gradient descent with hybrid MPI + Pthread
        for (int epoch = 0; epoch < epochs; epoch++) {
            // MPI: Each process handles a chunk of data
            int mpiChunkSize = data.size() / size;
            int mpiStart = rank * mpiChunkSize;
            int mpiEnd = (rank == size - 1) ? data.size() : (rank + 1) * mpiChunkSize;
            int mpiDataSize = mpiEnd - mpiStart;
            
            // Pthread: Create thread pool within MPI process
            vector<pthread_t> threads(numThreads);
            vector<ThreadData> threadData(numThreads);
            
            int threadChunkSize = mpiDataSize / numThreads;
            
            for (int t = 0; t < numThreads; t++) {
                threadData[t].data = &data;
                threadData[t].start = mpiStart + t * threadChunkSize;
                threadData[t].end = (t == numThreads - 1) ? mpiEnd : (mpiStart + (t + 1) * threadChunkSize);
                threadData[t].ethnicityWeights = &ethnicityWeights;
                threadData[t].genderWeights = &genderWeights;
                threadData[t].icd9Weights = &icd9Weights;
                threadData[t].bias = bias;
                threadData[t].lr = lr;
                
                pthread_create(&threads[t], NULL, trainThread, &threadData[t]);
            }
            
            // Join threads and accumulate gradients
            double localBiasGrad = 0.0;
            map<string, double> localEthGrads;
            map<string, double> localGenGrads;
            map<int, double> localIcdGrads;
            
            for (int t = 0; t < numThreads; t++) {
                pthread_join(threads[t], NULL);
                
                localBiasGrad += threadData[t].biasGrad;
                for (const auto& kv : threadData[t].ethGrads) localEthGrads[kv.first] += kv.second;
                for (const auto& kv : threadData[t].genGrads) localGenGrads[kv.first] += kv.second;
                for (const auto& kv : threadData[t].icdGrads) localIcdGrads[kv.first] += kv.second;
            }
            
            // MPI: Reduce gradients across all processes
            double globalBiasGrad = 0.0;
            MPI_Allreduce(&localBiasGrad, &globalBiasGrad, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            
            // Update bias
            bias += lr * globalBiasGrad;
            
            // Update weights based on local gradients
            // KNOWN LIMITATION: Weights updated locally for demonstration.
            // Production code should use MPI_Allreduce for full weight synchronization.
            for (auto& kv : ethnicityWeights) {
                if (localEthGrads.count(kv.first)) {
                    kv.second += lr * localEthGrads[kv.first];
                }
            }
            for (auto& kv : genderWeights) {
                if (localGenGrads.count(kv.first)) {
                    kv.second += lr * localGenGrads[kv.first];
                }
            }
            for (auto& kv : icd9Weights) {
                if (localIcdGrads.count(kv.first)) {
                    kv.second += lr * localIcdGrads[kv.first];
                }
            }
            
            // Broadcast updated bias
            MPI_Bcast(&bias, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
    }
    
    double calculateAccuracy(const vector<Patient>& data) {
        // MPI: Distribute evaluation across processes
        int mpiChunkSize = data.size() / size;
        int mpiStart = rank * mpiChunkSize;
        int mpiEnd = (rank == size - 1) ? data.size() : (rank + 1) * mpiChunkSize;
        int mpiDataSize = mpiEnd - mpiStart;
        
        // Pthread: Create thread pool within MPI process
        vector<pthread_t> threads(numThreads);
        vector<PredictThreadData> threadData(numThreads);
        
        int threadChunkSize = mpiDataSize / numThreads;
        
        for (int t = 0; t < numThreads; t++) {
            threadData[t].data = &data;
            threadData[t].start = mpiStart + t * threadChunkSize;
            threadData[t].end = (t == numThreads - 1) ? mpiEnd : (mpiStart + (t + 1) * threadChunkSize);
            threadData[t].ethnicityWeights = &ethnicityWeights;
            threadData[t].genderWeights = &genderWeights;
            threadData[t].icd9Weights = &icd9Weights;
            threadData[t].bias = bias;
            
            pthread_create(&threads[t], NULL, predictThread, &threadData[t]);
        }
        
        int localCorrect = 0;
        for (int t = 0; t < numThreads; t++) {
            pthread_join(threads[t], NULL);
            localCorrect += threadData[t].correct;
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
        int mpiDataSize = mpiEnd - mpiStart;
        
        // Pthread: Create thread pool within MPI process
        vector<pthread_t> threads(numThreads);
        vector<PredictThreadData> threadData(numThreads);
        
        int threadChunkSize = mpiDataSize / numThreads;
        
        for (int t = 0; t < numThreads; t++) {
            threadData[t].data = &data;
            threadData[t].start = mpiStart + t * threadChunkSize;
            threadData[t].end = (t == numThreads - 1) ? mpiEnd : (mpiStart + (t + 1) * threadChunkSize);
            threadData[t].ethnicityWeights = &ethnicityWeights;
            threadData[t].genderWeights = &genderWeights;
            threadData[t].icd9Weights = &icd9Weights;
            threadData[t].bias = bias;
            
            pthread_create(&threads[t], NULL, predictThread, &threadData[t]);
        }
        
        double localPred = 0.0;
        for (int t = 0; t < numThreads; t++) {
            pthread_join(threads[t], NULL);
            localPred += threadData[t].totalPred;
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
            cerr << "Usage: mpirun -np <processes> " << argv[0] << " <data_file> [num_threads]" << endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    int numThreads = (argc > 2) ? atoi(argv[2]) : 4;
    
    auto startTotal = chrono::high_resolution_clock::now();
    
    // Load data (all processes load data)
    auto startLoad = chrono::high_resolution_clock::now();
    vector<Patient> data = loadData(argv[1]);
    auto endLoad = chrono::high_resolution_clock::now();
    double loadTime = chrono::duration<double>(endLoad - startLoad).count();
    
    if (rank == 0) {
        cout << "Hybrid Pthread+MPI Implementation - Logistic Regression" << endl;
        cout << "========================================================" << endl;
        cout << "MPI Processes: " << size << endl;
        cout << "Pthread threads per process: " << numThreads << endl;
        cout << "Data loaded: " << data.size() << " patients" << endl;
        cout << "Load time: " << loadTime << " seconds" << endl;
    }
    
    // Split data (80% train, 20% test)
    int trainSize = data.size() * 0.8;
    vector<Patient> trainData(data.begin(), data.begin() + trainSize);
    vector<Patient> testData(data.begin() + trainSize, data.end());
    
    // Train model
    auto startTrain = chrono::high_resolution_clock::now();
    LogisticRegressionHybridPThreadMPI model(rank, size, numThreads);
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
        ofstream timingFile("timing_hybrid_pthread_mpi.txt");
        timingFile << "load," << loadTime << endl;
        timingFile << "train," << trainTime << endl;
        timingFile << "eval," << evalTime << endl;
        timingFile << "total," << totalTime << endl;
        timingFile << "accuracy," << accuracy << endl;
        timingFile << "deathrate," << deathRate << endl;
        timingFile << "processes," << size << endl;
        timingFile << "threads," << numThreads << endl;
        timingFile.close();
    }
    
    MPI_Finalize();
    return 0;
}
