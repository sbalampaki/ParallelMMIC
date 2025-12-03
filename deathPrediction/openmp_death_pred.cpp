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


using namespace std;

struct Patient {
    int subjectId;
    string ethnicity;
    string gender;
    int icd9Code1;
    int expireFlag;
};

class LogisticRegressionOMP {
private:
    map<string, double> ethnicityWeights;
    map<string, double> genderWeights;
    map<int, double> icd9Weights;
    double bias;
    
    double sigmoid(double z) {
        return 1.0 / (1.0 + exp(-z));
    }
    
public:
    LogisticRegressionOMP() : bias(0.0) {}
    
    void train(const vector<Patient>& data, int epochs = 100, double lr = 0.01) {
        // Initialize weights
        for (const auto& p : data) {
            ethnicityWeights[p.ethnicity] = 0.0;
            genderWeights[p.gender] = 0.0;
            icd9Weights[p.icd9Code1] = 0.0;
        }
        
        int numThreads = omp_get_max_threads();
        
        // Gradient descent with OpenMP parallelization
        for (int epoch = 0; epoch < epochs; epoch++) {
            double biasGrad = 0.0;
            map<string, double> ethGrads;
            map<string, double> genGrads;
            map<int, double> icdGrads;
            
            // Initialize gradient accumulators
            for (const auto& p : data) {
                ethGrads[p.ethnicity] = 0.0;
                genGrads[p.gender] = 0.0;
                icdGrads[p.icd9Code1] = 0.0;
            }
            
            // Parallel computation of gradients
            #pragma omp parallel
            {
                double local_bias = 0.0;
                map<string, double> local_eth;
                map<string, double> local_gen;
                map<int, double> local_icd;
                
                #pragma omp for
                for (size_t i = 0; i < data.size(); i++) {
                    const Patient& p = data[i];
                    double z = bias + ethnicityWeights[p.ethnicity] + 
                              genderWeights[p.gender] + icd9Weights[p.icd9Code1];
                    double pred = sigmoid(z);
                    double error = p.expireFlag - pred;
                    
                    local_bias += error;
                    local_eth[p.ethnicity] += error;
                    local_gen[p.gender] += error;
                    local_icd[p.icd9Code1] += error;
                }
                
                // Reduce gradients
                #pragma omp critical
                {
                    biasGrad += local_bias;
                    for (const auto& kv : local_eth) ethGrads[kv.first] += kv.second;
                    for (const auto& kv : local_gen) genGrads[kv.first] += kv.second;
                    for (const auto& kv : local_icd) icdGrads[kv.first] += kv.second;
                }
            }
            
            // Update weights
            bias += lr * biasGrad;
            for (auto& kv : ethnicityWeights) kv.second += lr * ethGrads[kv.first];
            for (auto& kv : genderWeights) kv.second += lr * genGrads[kv.first];
            for (auto& kv : icd9Weights) kv.second += lr * icdGrads[kv.first];
        }
    }
    
    double predict(const Patient& p) {
        double z = bias + ethnicityWeights[p.ethnicity] + 
                  genderWeights[p.gender] + icd9Weights[p.icd9Code1];
        return sigmoid(z);
    }
    
    double calculateAccuracy(const vector<Patient>& data) {
        int correct = 0;
        
        #pragma omp parallel for reduction(+:correct)
        for (size_t i = 0; i < data.size(); i++) {
            double pred = predict(data[i]);
            int predClass = pred >= 0.5 ? 1 : 0;
            if (predClass == data[i].expireFlag) correct++;
        }
        
        return (double)correct / data.size();
    }
    
    double calculateDeathRate(const vector<Patient>& data) {
        double totalPred = 0.0;
        
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
        cerr << "Usage: " << argv[0] << " <data_file>" << endl;
        return 1;
    }
    
    auto startTotal = chrono::high_resolution_clock::now();
    
    // Load data
    auto startLoad = chrono::high_resolution_clock::now();
    vector<Patient> data = loadData(argv[1]);
    auto endLoad = chrono::high_resolution_clock::now();
    double loadTime = chrono::duration<double>(endLoad - startLoad).count();
    
    int numThreads = omp_get_max_threads();
    
    cout << "OpenMP Implementation - Parallel Logistic Regression" << endl;
    cout << "====================================================" << endl;
    cout << "Number of threads: " << numThreads << endl;
    cout << "Data loaded: " << data.size() << " patients" << endl;
    cout << "Load time: " << loadTime << " seconds" << endl;
    
    // Split data (80% train, 20% test)
    int trainSize = data.size() * 0.8;
    vector<Patient> trainData(data.begin(), data.begin() + trainSize);
    vector<Patient> testData(data.begin() + trainSize, data.end());
    
    // Train model
    auto startTrain = chrono::high_resolution_clock::now();
    LogisticRegressionOMP model;
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
    ofstream timingFile("timing_openmp.txt");
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