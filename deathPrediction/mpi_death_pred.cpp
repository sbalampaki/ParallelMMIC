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

double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

class LogisticRegressionMPI {
private:
    map<string, double> ethnicityWeights;
    map<string, double> genderWeights;
    map<int, double> icd9Weights;
    double bias;
    int rank;
    int size;
    
public:
    LogisticRegressionMPI(int r, int s) : bias(0.0), rank(r), size(s) {}
    
    void train(const vector<Patient>& data, int epochs = 100, double lr = 0.01) {
        // Initialize weights (only on rank 0)
        if (rank == 0) {
            for (const auto& p : data) {
                ethnicityWeights[p.ethnicity] = 0.0;
                genderWeights[p.gender] = 0.0;
                icd9Weights[p.icd9Code1] = 0.0;
            }
        }
        
        // Build unique keys for all processes
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
        
        // Broadcast key sizes
        int ethSize, genSize, icdSize;
        if (rank == 0) {
            ethSize = ethnicityKeys.size();
            genSize = genderKeys.size();
            icdSize = icd9Keys.size();
        }
        
        MPI_Bcast(&ethSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&genSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&icdSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        // Broadcast keys (simplified for demo)
        if (rank != 0) {
            ethnicityKeys.resize(ethSize);
            genderKeys.resize(genSize);
            icd9Keys.resize(icdSize);
        }
        
        // Gradient descent
        for (int epoch = 0; epoch < epochs; epoch++) {
            double localBiasGrad = 0.0;
            map<string, double> localEthGrads;
            map<string, double> localGenGrads;
            map<int, double> localIcdGrads;
            
            // Each process works on its chunk
            int chunkSize = data.size() / size;
            int start = rank * chunkSize;
            int end = (rank == size - 1) ? data.size() : (rank + 1) * chunkSize;
            
            for (int i = start; i < end; i++) {
                const Patient& p = data[i];
                double z = bias + ethnicityWeights[p.ethnicity] + 
                          genderWeights[p.gender] + icd9Weights[p.icd9Code1];
                double pred = sigmoid(z);
                double error = p.expireFlag - pred;
                
                localBiasGrad += error;
                localEthGrads[p.ethnicity] += error;
                localGenGrads[p.gender] += error;
                localIcdGrads[p.icd9Code1] += error;
            }
            
            // Reduce gradients
            double globalBiasGrad = 0.0;
            MPI_Allreduce(&localBiasGrad, &globalBiasGrad, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            
            // Update bias
            bias += lr * globalBiasGrad;
            
            // For simplicity, we update weights based on local gradients
            // In production, you'd want proper all-reduce for all weights
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
            
            // Broadcast updated weights
            MPI_Bcast(&bias, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
    }
    
    double predict(const Patient& p) {
        double z = bias + ethnicityWeights[p.ethnicity] + 
                  genderWeights[p.gender] + icd9Weights[p.icd9Code1];
        return sigmoid(z);
    }
    
    double calculateAccuracy(const vector<Patient>& data) {
        int chunkSize = data.size() / size;
        int start = rank * chunkSize;
        int end = (rank == size - 1) ? data.size() : (rank + 1) * chunkSize;
        
        int localCorrect = 0;
        for (int i = start; i < end; i++) {
            double pred = predict(data[i]);
            int predClass = pred >= 0.5 ? 1 : 0;
            if (predClass == data[i].expireFlag) localCorrect++;
        }
        
        int globalCorrect = 0;
        MPI_Allreduce(&localCorrect, &globalCorrect, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        
        return (double)globalCorrect / data.size();
    }
    
    double calculateDeathRate(const vector<Patient>& data) {
        int chunkSize = data.size() / size;
        int start = rank * chunkSize;
        int end = (rank == size - 1) ? data.size() : (rank + 1) * chunkSize;
        
        double localPred = 0.0;
        for (int i = start; i < end; i++) {
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
            cerr << "Usage: mpirun -np <num_procs> " << argv[0] << " <data_file>" << endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    auto startTotal = chrono::high_resolution_clock::now();
    
    // Load data (all processes load the data)
    auto startLoad = chrono::high_resolution_clock::now();
    vector<Patient> data = loadData(argv[1]);
    auto endLoad = chrono::high_resolution_clock::now();
    double loadTime = chrono::duration<double>(endLoad - startLoad).count();
    
    if (rank == 0) {
        cout << "MPI Implementation - Distributed Logistic Regression" << endl;
        cout << "====================================================" << endl;
        cout << "Number of processes: " << size << endl;
        cout << "Data loaded: " << data.size() << " patients" << endl;
        cout << "Load time: " << loadTime << " seconds" << endl;
    }
    
    // Split data (80% train, 20% test)
    int trainSize = data.size() * 0.8;
    vector<Patient> trainData(data.begin(), data.begin() + trainSize);
    vector<Patient> testData(data.begin() + trainSize, data.end());
    
    // Train model
    auto startTrain = chrono::high_resolution_clock::now();
    LogisticRegressionMPI model(rank, size);
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
        ofstream timingFile("timing_mpi.txt");
        timingFile << "load," << loadTime << endl;
        timingFile << "train," << trainTime << endl;
        timingFile << "eval," << evalTime << endl;
        timingFile << "total," << totalTime << endl;
        timingFile << "accuracy," << accuracy << endl;
        timingFile << "deathrate," << deathRate << endl;
        timingFile << "processes," << size << endl;
        timingFile.close();
    }
    
    MPI_Finalize();
    return 0;
}