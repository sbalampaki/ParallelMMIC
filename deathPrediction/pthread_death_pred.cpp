#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <chrono>
#include <pthread.h>
#include <algorithm>

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

pthread_mutex_t updateMutex = PTHREAD_MUTEX_INITIALIZER;

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

class LogisticRegressionPThread {
private:
    map<string, double> ethnicityWeights;
    map<string, double> genderWeights;
    map<int, double> icd9Weights;
    double bias;
    int numThreads;
    
public:
    LogisticRegressionPThread(int threads = 4) : bias(0.0), numThreads(threads) {}
    
    void train(const vector<Patient>& data, int epochs = 100, double lr = 0.01) {
        // Initialize weights
        for (const auto& p : data) {
            ethnicityWeights[p.ethnicity] = 0.0;
            genderWeights[p.gender] = 0.0;
            icd9Weights[p.icd9Code1] = 0.0;
        }
        
        pthread_t threads[numThreads];
        ThreadData threadData[numThreads];
        
        int chunkSize = data.size() / numThreads;
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Create threads
            for (int t = 0; t < numThreads; t++) {
                threadData[t].data = &data;
                threadData[t].start = t * chunkSize;
                threadData[t].end = (t == numThreads - 1) ? data.size() : (t + 1) * chunkSize;
                threadData[t].ethnicityWeights = &ethnicityWeights;
                threadData[t].genderWeights = &genderWeights;
                threadData[t].icd9Weights = &icd9Weights;
                threadData[t].bias = bias;
                threadData[t].lr = lr;
                
                pthread_create(&threads[t], NULL, trainThread, &threadData[t]);
            }
            
            // Join threads and accumulate gradients
            double totalBiasGrad = 0.0;
            map<string, double> totalEthGrads;
            map<string, double> totalGenGrads;
            map<int, double> totalIcdGrads;
            
            for (int t = 0; t < numThreads; t++) {
                pthread_join(threads[t], NULL);
                
                totalBiasGrad += threadData[t].biasGrad;
                for (const auto& kv : threadData[t].ethGrads) totalEthGrads[kv.first] += kv.second;
                for (const auto& kv : threadData[t].genGrads) totalGenGrads[kv.first] += kv.second;
                for (const auto& kv : threadData[t].icdGrads) totalIcdGrads[kv.first] += kv.second;
            }
            
            // Update weights
            bias += lr * totalBiasGrad;
            for (auto& kv : ethnicityWeights) kv.second += lr * totalEthGrads[kv.first];
            for (auto& kv : genderWeights) kv.second += lr * totalGenGrads[kv.first];
            for (auto& kv : icd9Weights) kv.second += lr * totalIcdGrads[kv.first];
        }
    }
    
    double calculateAccuracy(const vector<Patient>& data) {
        pthread_t threads[numThreads];
        PredictThreadData threadData[numThreads];
        
        int chunkSize = data.size() / numThreads;
        
        for (int t = 0; t < numThreads; t++) {
            threadData[t].data = &data;
            threadData[t].start = t * chunkSize;
            threadData[t].end = (t == numThreads - 1) ? data.size() : (t + 1) * chunkSize;
            threadData[t].ethnicityWeights = &ethnicityWeights;
            threadData[t].genderWeights = &genderWeights;
            threadData[t].icd9Weights = &icd9Weights;
            threadData[t].bias = bias;
            
            pthread_create(&threads[t], NULL, predictThread, &threadData[t]);
        }
        
        int totalCorrect = 0;
        for (int t = 0; t < numThreads; t++) {
            pthread_join(threads[t], NULL);
            totalCorrect += threadData[t].correct;
        }
        
        return (double)totalCorrect / data.size();
    }
    
    double calculateDeathRate(const vector<Patient>& data) {
        pthread_t threads[numThreads];
        PredictThreadData threadData[numThreads];
        
        int chunkSize = data.size() / numThreads;
        
        for (int t = 0; t < numThreads; t++) {
            threadData[t].data = &data;
            threadData[t].start = t * chunkSize;
            threadData[t].end = (t == numThreads - 1) ? data.size() : (t + 1) * chunkSize;
            threadData[t].ethnicityWeights = &ethnicityWeights;
            threadData[t].genderWeights = &genderWeights;
            threadData[t].icd9Weights = &icd9Weights;
            threadData[t].bias = bias;
            
            pthread_create(&threads[t], NULL, predictThread, &threadData[t]);
        }
        
        double totalPred = 0.0;
        for (int t = 0; t < numThreads; t++) {
            pthread_join(threads[t], NULL);
            totalPred += threadData[t].totalPred;
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
        cerr << "Usage: " << argv[0] << " <data_file> [num_threads]" << endl;
        return 1;
    }
    
    int numThreads = argc > 2 ? atoi(argv[2]) : 4;
    
    auto startTotal = chrono::high_resolution_clock::now();
    
    // Load data
    auto startLoad = chrono::high_resolution_clock::now();
    vector<Patient> data = loadData(argv[1]);
    auto endLoad = chrono::high_resolution_clock::now();
    double loadTime = chrono::duration<double>(endLoad - startLoad).count();
    
    cout << "Pthreads Implementation - Parallel Logistic Regression" << endl;
    cout << "======================================================" << endl;
    cout << "Number of threads: " << numThreads << endl;
    cout << "Data loaded: " << data.size() << " patients" << endl;
    cout << "Load time: " << loadTime << " seconds" << endl;
    
    // Split data (80% train, 20% test)
    int trainSize = data.size() * 0.8;
    vector<Patient> trainData(data.begin(), data.begin() + trainSize);
    vector<Patient> testData(data.begin() + trainSize, data.end());
    
    // Train model
    auto startTrain = chrono::high_resolution_clock::now();
    LogisticRegressionPThread model(numThreads);
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
    ofstream timingFile("timing_pthread.txt");
    timingFile << "load," << loadTime << endl;
    timingFile << "train," << trainTime << endl;
    timingFile << "eval," << evalTime << endl;
    timingFile << "total," << totalTime << endl;
    timingFile << "accuracy," << accuracy << endl;
    timingFile << "deathrate," << deathRate << endl;
    timingFile << "threads," << numThreads << endl;
    timingFile.close();
    
    return 0;
}