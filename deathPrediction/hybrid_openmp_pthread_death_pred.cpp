#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <chrono>
#include <algorithm>
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

class LogisticRegressionHybridOMPPThread {
private:
    map<string, double> ethnicityWeights;
    map<string, double> genderWeights;
    map<int, double> icd9Weights;
    double bias;
    int numPThreads;
    
public:
    LogisticRegressionHybridOMPPThread(int pthreads = 2) : bias(0.0), numPThreads(pthreads) {}
    
    void train(const vector<Patient>& data, int epochs = 100, double lr = 0.01) {
        // Initialize weights
        for (const auto& p : data) {
            ethnicityWeights[p.ethnicity] = 0.0;
            genderWeights[p.gender] = 0.0;
            icd9Weights[p.icd9Code1] = 0.0;
        }
        
        // Gradient descent with hybrid OpenMP + Pthread
        for (int epoch = 0; epoch < epochs; epoch++) {
            // OpenMP: Parallelize outer loop over batches
            double totalBiasGrad = 0.0;
            map<string, double> totalEthGrads;
            map<string, double> totalGenGrads;
            map<int, double> totalIcdGrads;
            
            // Divide data into OpenMP chunks
            int ompChunks = omp_get_max_threads();
            int ompChunkSize = data.size() / ompChunks;
            
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                int ompStart = tid * ompChunkSize;
                int ompEnd = (tid == ompChunks - 1) ? data.size() : (tid + 1) * ompChunkSize;
                int ompDataSize = ompEnd - ompStart;
                
                // Pthread: Fine-grained parallelism within each OpenMP thread
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
                double localBiasGrad = 0.0;
                map<string, double> localEthGrads;
                map<string, double> localGenGrads;
                map<int, double> localIcdGrads;
                
                for (int t = 0; t < numPThreads; t++) {
                    pthread_join(threads[t], NULL);
                    
                    localBiasGrad += threadData[t].biasGrad;
                    for (const auto& kv : threadData[t].ethGrads) localEthGrads[kv.first] += kv.second;
                    for (const auto& kv : threadData[t].genGrads) localGenGrads[kv.first] += kv.second;
                    for (const auto& kv : threadData[t].icdGrads) localIcdGrads[kv.first] += kv.second;
                }
                
                // OpenMP reduction: Combine results from all OpenMP threads
                #pragma omp critical
                {
                    totalBiasGrad += localBiasGrad;
                    for (const auto& kv : localEthGrads) totalEthGrads[kv.first] += kv.second;
                    for (const auto& kv : localGenGrads) totalGenGrads[kv.first] += kv.second;
                    for (const auto& kv : localIcdGrads) totalIcdGrads[kv.first] += kv.second;
                }
            }
            
            // Update weights
            bias += lr * totalBiasGrad;
            for (auto& kv : ethnicityWeights) kv.second += lr * totalEthGrads[kv.first];
            for (auto& kv : genderWeights) kv.second += lr * totalGenGrads[kv.first];
            for (auto& kv : icd9Weights) kv.second += lr * totalIcdGrads[kv.first];
        }
    }
    
    double predict(const Patient& p) {
        double z = bias + ethnicityWeights[p.ethnicity] + 
                  genderWeights[p.gender] + icd9Weights[p.icd9Code1];
        return sigmoid(z);
    }
    
    double calculateAccuracy(const vector<Patient>& data) {
        int totalCorrect = 0;
        
        // OpenMP: Parallelize evaluation
        #pragma omp parallel for reduction(+:totalCorrect)
        for (size_t i = 0; i < data.size(); i++) {
            double pred = predict(data[i]);
            int predClass = pred >= 0.5 ? 1 : 0;
            if (predClass == data[i].expireFlag) totalCorrect++;
        }
        
        return (double)totalCorrect / data.size();
    }
    
    double calculateDeathRate(const vector<Patient>& data) {
        double totalPred = 0.0;
        
        // OpenMP: Parallelize prediction
        #pragma omp parallel for reduction(+:totalPred)
        for (size_t i = 0; i < data.size(); i++) {
            totalPred += predict(data[i]);
        }
        
        return totalPred / data.size();
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
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <data_file> [num_pthreads]" << endl;
        return 1;
    }
    
    int numPThreads = (argc > 2) ? atoi(argv[2]) : 2;
    
    auto startTotal = chrono::high_resolution_clock::now();
    
    // Load data
    auto startLoad = chrono::high_resolution_clock::now();
    vector<Patient> data = loadData(argv[1]);
    auto endLoad = chrono::high_resolution_clock::now();
    double loadTime = chrono::duration<double>(endLoad - startLoad).count();
    
    cout << "Hybrid OpenMP+Pthread Implementation - Logistic Regression" << endl;
    cout << "===========================================================" << endl;
    cout << "OpenMP Threads: " << omp_get_max_threads() << endl;
    cout << "Pthreads per OpenMP thread: " << numPThreads << endl;
    cout << "Total parallelism: " << (omp_get_max_threads() * numPThreads) << " threads" << endl;
    cout << "Data loaded: " << data.size() << " patients" << endl;
    cout << "Load time: " << loadTime << " seconds" << endl;
    
    // Split data (80% train, 20% test)
    int trainSize = data.size() * 0.8;
    vector<Patient> trainData(data.begin(), data.begin() + trainSize);
    vector<Patient> testData(data.begin() + trainSize, data.end());
    
    // Train model
    auto startTrain = chrono::high_resolution_clock::now();
    LogisticRegressionHybridOMPPThread model(numPThreads);
    model.train(trainData, 100, 0.01);
    auto endTrain = chrono::high_resolution_clock::now();
    double trainTime = chrono::duration<double>(endTrain - startTrain).count();
    
    cout << "Training time: " << trainTime << " seconds" << endl;
    
    // Evaluate
    auto startEval = chrono::high_resolution_clock::now();
    double accuracy = model.calculateAccuracy(testData);
    double deathRate = model.calculateDeathRate(testData);
    auto endEval = chrono::high_resolution_clock::now();
    double evalTime = chrono::duration<double>(endEval - startEval).count();
    
    auto endTotal = chrono::high_resolution_clock::now();
    double totalTime = chrono::duration<double>(endTotal - startTotal).count();
    
    cout << "Evaluation time: " << evalTime << " seconds" << endl;
    cout << "Total execution time: " << totalTime << " seconds" << endl;
    cout << "\nResults:" << endl;
    cout << "Accuracy: " << (accuracy * 100) << "%" << endl;
    cout << "Predicted Death Rate: " << (deathRate * 100) << "%" << endl;
    
    // Save timing to file
    ofstream timingFile("timing_hybrid_openmp_pthread.txt");
    timingFile << "load," << loadTime << endl;
    timingFile << "train," << trainTime << endl;
    timingFile << "eval," << evalTime << endl;
    timingFile << "total," << totalTime << endl;
    timingFile << "accuracy," << accuracy << endl;
    timingFile << "deathrate," << deathRate << endl;
    timingFile << "threads," << (omp_get_max_threads() * numPThreads) << endl;
    timingFile.close();
    
    return 0;
}
