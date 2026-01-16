#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>

using namespace std;

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA error in " << __FILE__ << " line " << __LINE__ << ": " \
                 << cudaGetErrorString(err) << endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

struct Patient {
    int subjectId;
    string ethnicity;
    string gender;
    int icd9Code1;
    int expireFlag;
};

// CUDA kernel for sigmoid function
__device__ double sigmoid_device(double z) {
    return 1.0 / (1.0 + exp(-z));
}

// CUDA kernel for computing predictions
__global__ void computePredictions(
    const int* ethnicityIndices,
    const int* genderIndices,
    const int* icd9Indices,
    const double* ethnicityWeights,
    const double* genderWeights,
    const double* icd9Weights,
    double bias,
    double* predictions,
    int n,
    int numEthnicities,
    int numGenders,
    int numIcd9
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int ethIdx = ethnicityIndices[idx];
        int genIdx = genderIndices[idx];
        int icdIdx = icd9Indices[idx];
        
        double z = bias;
        if (ethIdx >= 0 && ethIdx < numEthnicities) z += ethnicityWeights[ethIdx];
        if (genIdx >= 0 && genIdx < numGenders) z += genderWeights[genIdx];
        if (icdIdx >= 0 && icdIdx < numIcd9) z += icd9Weights[icdIdx];
        
        predictions[idx] = sigmoid_device(z);
    }
}

// CUDA kernel for computing gradients
__global__ void computeGradients(
    const int* ethnicityIndices,
    const int* genderIndices,
    const int* icd9Indices,
    const int* labels,
    const double* predictions,
    double* ethnicityGrads,
    double* genderGrads,
    double* icd9Grads,
    double* biasGrad,
    int n,
    int numEthnicities,
    int numGenders,
    int numIcd9
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double error = labels[idx] - predictions[idx];
        
        int ethIdx = ethnicityIndices[idx];
        int genIdx = genderIndices[idx];
        int icdIdx = icd9Indices[idx];
        
        // Use atomic operations for gradient accumulation
        if (ethIdx >= 0 && ethIdx < numEthnicities) {
            atomicAdd(&ethnicityGrads[ethIdx], error);
        }
        if (genIdx >= 0 && genIdx < numGenders) {
            atomicAdd(&genderGrads[genIdx], error);
        }
        if (icdIdx >= 0 && icdIdx < numIcd9) {
            atomicAdd(&icd9Grads[icdIdx], error);
        }
        atomicAdd(biasGrad, error);
    }
}

// CUDA kernel for updating weights
__global__ void updateWeights(
    double* weights,
    const double* gradients,
    double learningRate,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        weights[idx] += learningRate * gradients[idx];
    }
}

// CUDA kernel for computing accuracy
__global__ void computeAccuracy(
    const double* predictions,
    const int* labels,
    int* correct,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int predClass = predictions[idx] >= 0.5 ? 1 : 0;
        if (predClass == labels[idx]) {
            atomicAdd(correct, 1);
        }
    }
}

class LogisticRegressionCUDA {
private:
    map<string, int> ethnicityMap;
    map<string, int> genderMap;
    map<int, int> icd9Map;
    
    vector<double> ethnicityWeights;
    vector<double> genderWeights;
    vector<double> icd9Weights;
    double bias;
    
    int numEthnicities;
    int numGenders;
    int numIcd9;
    
public:
    LogisticRegressionCUDA() : bias(0.0), numEthnicities(0), numGenders(0), numIcd9(0) {}
    
    void train(const vector<Patient>& data, int epochs = 100, double lr = 0.01) {
        int n = data.size();
        
        // Build mappings and initialize weights
        int ethCounter = 0, genCounter = 0, icdCounter = 0;
        for (const auto& p : data) {
            if (ethnicityMap.find(p.ethnicity) == ethnicityMap.end()) {
                ethnicityMap[p.ethnicity] = ethCounter++;
            }
            if (genderMap.find(p.gender) == genderMap.end()) {
                genderMap[p.gender] = genCounter++;
            }
            if (icd9Map.find(p.icd9Code1) == icd9Map.end()) {
                icd9Map[p.icd9Code1] = icdCounter++;
            }
        }
        
        numEthnicities = ethCounter;
        numGenders = genCounter;
        numIcd9 = icdCounter;
        
        ethnicityWeights.resize(numEthnicities, 0.0);
        genderWeights.resize(numGenders, 0.0);
        icd9Weights.resize(numIcd9, 0.0);
        
        // Prepare data arrays
        vector<int> ethnicityIndices(n);
        vector<int> genderIndices(n);
        vector<int> icd9Indices(n);
        vector<int> labels(n);
        
        for (int i = 0; i < n; i++) {
            ethnicityIndices[i] = ethnicityMap[data[i].ethnicity];
            genderIndices[i] = genderMap[data[i].gender];
            icd9Indices[i] = icd9Map[data[i].icd9Code1];
            labels[i] = data[i].expireFlag;
        }
        
        // Allocate device memory
        int *d_ethnicityIndices, *d_genderIndices, *d_icd9Indices, *d_labels;
        double *d_ethnicityWeights, *d_genderWeights, *d_icd9Weights;
        double *d_ethnicityGrads, *d_genderGrads, *d_icd9Grads;
        double *d_predictions, *d_biasGrad;
        
        CUDA_CHECK(cudaMalloc(&d_ethnicityIndices, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_genderIndices, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_icd9Indices, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_labels, n * sizeof(int)));
        
        CUDA_CHECK(cudaMalloc(&d_ethnicityWeights, numEthnicities * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_genderWeights, numGenders * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_icd9Weights, numIcd9 * sizeof(double)));
        
        CUDA_CHECK(cudaMalloc(&d_ethnicityGrads, numEthnicities * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_genderGrads, numGenders * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_icd9Grads, numIcd9 * sizeof(double)));
        
        CUDA_CHECK(cudaMalloc(&d_predictions, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_biasGrad, sizeof(double)));
        
        // Copy data to device
        CUDA_CHECK(cudaMemcpy(d_ethnicityIndices, ethnicityIndices.data(), 
                             n * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_genderIndices, genderIndices.data(), 
                             n * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_icd9Indices, icd9Indices.data(), 
                             n * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_labels, labels.data(), 
                             n * sizeof(int), cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaMemcpy(d_ethnicityWeights, ethnicityWeights.data(), 
                             numEthnicities * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_genderWeights, genderWeights.data(), 
                             numGenders * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_icd9Weights, icd9Weights.data(), 
                             numIcd9 * sizeof(double), cudaMemcpyHostToDevice));
        
        // Training loop
        int blockSize = 256;
        int numBlocks = (n + blockSize - 1) / blockSize;
        int weightBlocksEth = (numEthnicities + blockSize - 1) / blockSize;
        int weightBlocksGen = (numGenders + blockSize - 1) / blockSize;
        int weightBlocksIcd = (numIcd9 + blockSize - 1) / blockSize;
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Zero gradients
            CUDA_CHECK(cudaMemset(d_ethnicityGrads, 0, numEthnicities * sizeof(double)));
            CUDA_CHECK(cudaMemset(d_genderGrads, 0, numGenders * sizeof(double)));
            CUDA_CHECK(cudaMemset(d_icd9Grads, 0, numIcd9 * sizeof(double)));
            CUDA_CHECK(cudaMemset(d_biasGrad, 0, sizeof(double)));
            
            // Compute predictions
            computePredictions<<<numBlocks, blockSize>>>(
                d_ethnicityIndices, d_genderIndices, d_icd9Indices,
                d_ethnicityWeights, d_genderWeights, d_icd9Weights,
                bias, d_predictions, n,
                numEthnicities, numGenders, numIcd9
            );
            CUDA_CHECK(cudaGetLastError());
            
            // Compute gradients
            computeGradients<<<numBlocks, blockSize>>>(
                d_ethnicityIndices, d_genderIndices, d_icd9Indices,
                d_labels, d_predictions,
                d_ethnicityGrads, d_genderGrads, d_icd9Grads, d_biasGrad,
                n, numEthnicities, numGenders, numIcd9
            );
            CUDA_CHECK(cudaGetLastError());
            
            // Update weights
            if (numEthnicities > 0) {
                updateWeights<<<weightBlocksEth, blockSize>>>(
                    d_ethnicityWeights, d_ethnicityGrads, lr, numEthnicities
                );
            }
            if (numGenders > 0) {
                updateWeights<<<weightBlocksGen, blockSize>>>(
                    d_genderWeights, d_genderGrads, lr, numGenders
                );
            }
            if (numIcd9 > 0) {
                updateWeights<<<weightBlocksIcd, blockSize>>>(
                    d_icd9Weights, d_icd9Grads, lr, numIcd9
                );
            }
            CUDA_CHECK(cudaGetLastError());
            
            // Update bias
            double biasGradHost;
            CUDA_CHECK(cudaMemcpy(&biasGradHost, d_biasGrad, 
                                 sizeof(double), cudaMemcpyDeviceToHost));
            bias += lr * biasGradHost;
        }
        
        // Copy weights back to host
        CUDA_CHECK(cudaMemcpy(ethnicityWeights.data(), d_ethnicityWeights, 
                             numEthnicities * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(genderWeights.data(), d_genderWeights, 
                             numGenders * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(icd9Weights.data(), d_icd9Weights, 
                             numIcd9 * sizeof(double), cudaMemcpyDeviceToHost));
        
        // Free device memory
        CUDA_CHECK(cudaFree(d_ethnicityIndices));
        CUDA_CHECK(cudaFree(d_genderIndices));
        CUDA_CHECK(cudaFree(d_icd9Indices));
        CUDA_CHECK(cudaFree(d_labels));
        CUDA_CHECK(cudaFree(d_ethnicityWeights));
        CUDA_CHECK(cudaFree(d_genderWeights));
        CUDA_CHECK(cudaFree(d_icd9Weights));
        CUDA_CHECK(cudaFree(d_ethnicityGrads));
        CUDA_CHECK(cudaFree(d_genderGrads));
        CUDA_CHECK(cudaFree(d_icd9Grads));
        CUDA_CHECK(cudaFree(d_predictions));
        CUDA_CHECK(cudaFree(d_biasGrad));
    }
    
    double predict(const Patient& p) {
        double z = bias;
        auto ethIt = ethnicityMap.find(p.ethnicity);
        auto genIt = genderMap.find(p.gender);
        auto icdIt = icd9Map.find(p.icd9Code1);
        
        if (ethIt != ethnicityMap.end()) {
            z += ethnicityWeights[ethIt->second];
        }
        if (genIt != genderMap.end()) {
            z += genderWeights[genIt->second];
        }
        if (icdIt != icd9Map.end()) {
            z += icd9Weights[icdIt->second];
        }
        
        return 1.0 / (1.0 + exp(-z));
    }
    
    double calculateAccuracy(const vector<Patient>& data) {
        int n = data.size();
        vector<int> ethnicityIndices(n);
        vector<int> genderIndices(n);
        vector<int> icd9Indices(n);
        vector<int> labels(n);
        
        for (int i = 0; i < n; i++) {
            auto ethIt = ethnicityMap.find(data[i].ethnicity);
            auto genIt = genderMap.find(data[i].gender);
            auto icdIt = icd9Map.find(data[i].icd9Code1);
            
            ethnicityIndices[i] = (ethIt != ethnicityMap.end()) ? ethIt->second : -1;
            genderIndices[i] = (genIt != genderMap.end()) ? genIt->second : -1;
            icd9Indices[i] = (icdIt != icd9Map.end()) ? icdIt->second : -1;
            labels[i] = data[i].expireFlag;
        }
        
        // Allocate device memory
        int *d_ethnicityIndices, *d_genderIndices, *d_icd9Indices, *d_labels, *d_correct;
        double *d_ethnicityWeights, *d_genderWeights, *d_icd9Weights, *d_predictions;
        
        CUDA_CHECK(cudaMalloc(&d_ethnicityIndices, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_genderIndices, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_icd9Indices, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_labels, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_correct, sizeof(int)));
        
        CUDA_CHECK(cudaMalloc(&d_ethnicityWeights, numEthnicities * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_genderWeights, numGenders * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_icd9Weights, numIcd9 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_predictions, n * sizeof(double)));
        
        // Copy data to device
        CUDA_CHECK(cudaMemcpy(d_ethnicityIndices, ethnicityIndices.data(), 
                             n * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_genderIndices, genderIndices.data(), 
                             n * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_icd9Indices, icd9Indices.data(), 
                             n * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_labels, labels.data(), 
                             n * sizeof(int), cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaMemcpy(d_ethnicityWeights, ethnicityWeights.data(), 
                             numEthnicities * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_genderWeights, genderWeights.data(), 
                             numGenders * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_icd9Weights, icd9Weights.data(), 
                             numIcd9 * sizeof(double), cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaMemset(d_correct, 0, sizeof(int)));
        
        // Compute predictions and accuracy
        int blockSize = 256;
        int numBlocks = (n + blockSize - 1) / blockSize;
        
        computePredictions<<<numBlocks, blockSize>>>(
            d_ethnicityIndices, d_genderIndices, d_icd9Indices,
            d_ethnicityWeights, d_genderWeights, d_icd9Weights,
            bias, d_predictions, n,
            numEthnicities, numGenders, numIcd9
        );
        CUDA_CHECK(cudaGetLastError());
        
        computeAccuracy<<<numBlocks, blockSize>>>(
            d_predictions, d_labels, d_correct, n
        );
        CUDA_CHECK(cudaGetLastError());
        
        // Copy result back
        int correctHost;
        CUDA_CHECK(cudaMemcpy(&correctHost, d_correct, 
                             sizeof(int), cudaMemcpyDeviceToHost));
        
        // Free device memory
        CUDA_CHECK(cudaFree(d_ethnicityIndices));
        CUDA_CHECK(cudaFree(d_genderIndices));
        CUDA_CHECK(cudaFree(d_icd9Indices));
        CUDA_CHECK(cudaFree(d_labels));
        CUDA_CHECK(cudaFree(d_correct));
        CUDA_CHECK(cudaFree(d_ethnicityWeights));
        CUDA_CHECK(cudaFree(d_genderWeights));
        CUDA_CHECK(cudaFree(d_icd9Weights));
        CUDA_CHECK(cudaFree(d_predictions));
        
        return (double)correctHost / n;
    }
    
    double calculateDeathRate(const vector<Patient>& data) {
        double totalPred = 0.0;
        for (const auto& p : data) {
            totalPred += predict(p);
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
            
            // SUBJECT_ID
            if (!getline(ss, temp, ',')) continue;
            p.subjectId = safeStoi(temp);
            
            // HADM_ID (skip)
            if (!getline(ss, temp, ',')) continue;
            
            // HOSPITAL (skip)
            if (!getline(ss, temp, ',')) continue;
            
            // EXPIRE_FLAG
            if (!getline(ss, temp, ',')) continue;
            p.expireFlag = safeStoi(temp);
            
            // ADMITTIME (skip)
            if (!getline(ss, temp, ',')) continue;
            
            // ETHNICITY
            if (!getline(ss, p.ethnicity, ',')) continue;
            p.ethnicity.erase(0, p.ethnicity.find_first_not_of(" \t\r\n"));
            p.ethnicity.erase(p.ethnicity.find_last_not_of(" \t\r\n") + 1);
            
            // GENDER
            if (!getline(ss, p.gender, ',')) continue;
            p.gender.erase(0, p.gender.find_first_not_of(" \t\r\n"));
            p.gender.erase(p.gender.find_last_not_of(" \t\r\n") + 1);
            
            // DOB (skip)
            if (!getline(ss, temp, ',')) continue;
            
            // AGE (skip)
            if (!getline(ss, temp, ',')) continue;
            
            // ICD9_CODE_1
            if (!getline(ss, temp, ',')) {
                p.icd9Code1 = 0;
            } else {
                p.icd9Code1 = safeStoi(temp);
            }
            
            patients.push_back(p);
        } catch (const exception& e) {
            cerr << "Warning: Error parsing line " << lineNum << ": " << e.what() << endl;
            continue;
        }
    }
    
    if (patients.empty()) {
        cerr << "Error: No valid data loaded from file!" << endl;
    }
    
    return patients;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <data_file>" << endl;
        return 1;
    }
    
    // Check for CUDA device
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        cerr << "Error: No CUDA-capable device found!" << endl;
        return 1;
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    cout << "CUDA Implementation - Logistic Regression" << endl;
    cout << "==========================================" << endl;
    cout << "Using GPU: " << prop.name << endl;
    cout << "Compute Capability: " << prop.major << "." << prop.minor << endl;
    
    auto startTotal = chrono::high_resolution_clock::now();
    
    // Load data
    auto startLoad = chrono::high_resolution_clock::now();
    vector<Patient> data = loadData(argv[1]);
    auto endLoad = chrono::high_resolution_clock::now();
    double loadTime = chrono::duration<double>(endLoad - startLoad).count();
    
    cout << "Data loaded: " << data.size() << " patients" << endl;
    cout << "Load time: " << loadTime << " seconds" << endl;
    
    // Split data (80% train, 20% test)
    int trainSize = data.size() * 0.8;
    vector<Patient> trainData(data.begin(), data.begin() + trainSize);
    vector<Patient> testData(data.begin() + trainSize, data.end());
    
    // Train model
    auto startTrain = chrono::high_resolution_clock::now();
    LogisticRegressionCUDA model;
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
    ofstream timingFile("timing_cuda.txt");
    timingFile << "load," << loadTime << endl;
    timingFile << "train," << trainTime << endl;
    timingFile << "eval," << evalTime << endl;
    timingFile << "total," << totalTime << endl;
    timingFile << "accuracy," << accuracy << endl;
    timingFile << "deathrate," << deathRate << endl;
    timingFile.close();
    
    return 0;
}
