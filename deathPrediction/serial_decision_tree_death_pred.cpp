#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <limits>

using namespace std;

struct Patient {
    int subjectId;
    string ethnicity;
    string gender;
    int icd9Code1;
    int expireFlag;
};

// Decision Tree Node
struct TreeNode {
    bool isLeaf;
    int prediction;  // For leaf nodes
    
    // For internal nodes
    string splitFeature;  // "ethnicity", "gender", or "icd9"
    string splitValueStr;  // For categorical (ethnicity, gender)
    int splitValueInt;     // For ICD9 code
    
    TreeNode* left;
    TreeNode* right;
    
    TreeNode() : isLeaf(false), prediction(0), splitValueInt(0), left(nullptr), right(nullptr) {}
    ~TreeNode() {
        delete left;
        delete right;
    }
};

class DecisionTree {
private:
    TreeNode* root;
    int maxDepth;
    int minSamplesSplit;
    
    // Calculate entropy
    double calculateEntropy(const vector<Patient>& data) {
        if (data.empty()) return 0.0;
        
        int positive = 0;
        for (const auto& p : data) {
            if (p.expireFlag == 1) positive++;
        }
        
        if (positive == 0 || positive == (int)data.size()) return 0.0;
        
        double p1 = (double)positive / data.size();
        double p0 = 1.0 - p1;
        
        return -p1 * log2(p1) - p0 * log2(p0);
    }
    
    // Calculate information gain for a split
    double calculateInformationGain(const vector<Patient>& data, 
                                     const vector<Patient>& left, 
                                     const vector<Patient>& right) {
        double parentEntropy = calculateEntropy(data);
        double leftWeight = (double)left.size() / data.size();
        double rightWeight = (double)right.size() / data.size();
        
        return parentEntropy - (leftWeight * calculateEntropy(left) + 
                                rightWeight * calculateEntropy(right));
    }
    
    // Find best split
    void findBestSplit(const vector<Patient>& data, 
                       string& bestFeature, 
                       string& bestValueStr, 
                       int& bestValueInt,
                       double& bestGain) {
        bestGain = 0.0;
        bestFeature = "";
        
        // Try splitting on ethnicity
        map<string, vector<Patient>> ethnicityGroups;
        for (const auto& p : data) {
            ethnicityGroups[p.ethnicity].push_back(p);
        }
        
        for (const auto& pair : ethnicityGroups) {
            vector<Patient> left, right;
            for (const auto& p : data) {
                if (p.ethnicity == pair.first) {
                    left.push_back(p);
                } else {
                    right.push_back(p);
                }
            }
            
            if (!left.empty() && !right.empty()) {
                double gain = calculateInformationGain(data, left, right);
                if (gain > bestGain) {
                    bestGain = gain;
                    bestFeature = "ethnicity";
                    bestValueStr = pair.first;
                }
            }
        }
        
        // Try splitting on gender
        map<string, vector<Patient>> genderGroups;
        for (const auto& p : data) {
            genderGroups[p.gender].push_back(p);
        }
        
        for (const auto& pair : genderGroups) {
            vector<Patient> left, right;
            for (const auto& p : data) {
                if (p.gender == pair.first) {
                    left.push_back(p);
                } else {
                    right.push_back(p);
                }
            }
            
            if (!left.empty() && !right.empty()) {
                double gain = calculateInformationGain(data, left, right);
                if (gain > bestGain) {
                    bestGain = gain;
                    bestFeature = "gender";
                    bestValueStr = pair.first;
                }
            }
        }
        
        // Try splitting on ICD9 code
        map<int, int> icd9Counts;
        for (const auto& p : data) {
            icd9Counts[p.icd9Code1]++;
        }
        
        for (const auto& pair : icd9Counts) {
            vector<Patient> left, right;
            for (const auto& p : data) {
                if (p.icd9Code1 == pair.first) {
                    left.push_back(p);
                } else {
                    right.push_back(p);
                }
            }
            
            if (!left.empty() && !right.empty()) {
                double gain = calculateInformationGain(data, left, right);
                if (gain > bestGain) {
                    bestGain = gain;
                    bestFeature = "icd9";
                    bestValueInt = pair.first;
                }
            }
        }
    }
    
    // Build tree recursively
    TreeNode* buildTree(const vector<Patient>& data, int depth) {
        TreeNode* node = new TreeNode();
        
        // Check stopping criteria
        if (data.empty()) {
            node->isLeaf = true;
            node->prediction = 0;
            return node;
        }
        
        // Check if all samples have same label
        int positive = 0;
        for (const auto& p : data) {
            if (p.expireFlag == 1) positive++;
        }
        
        if (positive == 0 || positive == (int)data.size() || 
            depth >= maxDepth || (int)data.size() < minSamplesSplit) {
            node->isLeaf = true;
            node->prediction = (positive > (int)data.size() / 2) ? 1 : 0;
            return node;
        }
        
        // Find best split
        string bestFeature, bestValueStr;
        int bestValueInt = 0;
        double bestGain = 0.0;
        
        findBestSplit(data, bestFeature, bestValueStr, bestValueInt, bestGain);
        
        // If no good split found, make leaf
        if (bestGain <= 0.0 || bestFeature.empty()) {
            node->isLeaf = true;
            node->prediction = (positive > (int)data.size() / 2) ? 1 : 0;
            return node;
        }
        
        // Split data
        vector<Patient> left, right;
        if (bestFeature == "ethnicity") {
            for (const auto& p : data) {
                if (p.ethnicity == bestValueStr) {
                    left.push_back(p);
                } else {
                    right.push_back(p);
                }
            }
            node->splitFeature = "ethnicity";
            node->splitValueStr = bestValueStr;
        } else if (bestFeature == "gender") {
            for (const auto& p : data) {
                if (p.gender == bestValueStr) {
                    left.push_back(p);
                } else {
                    right.push_back(p);
                }
            }
            node->splitFeature = "gender";
            node->splitValueStr = bestValueStr;
        } else { // icd9
            for (const auto& p : data) {
                if (p.icd9Code1 == bestValueInt) {
                    left.push_back(p);
                } else {
                    right.push_back(p);
                }
            }
            node->splitFeature = "icd9";
            node->splitValueInt = bestValueInt;
        }
        
        // Build subtrees
        node->left = buildTree(left, depth + 1);
        node->right = buildTree(right, depth + 1);
        
        return node;
    }
    
    // Predict for single patient
    int predictOne(TreeNode* node, const Patient& p) {
        if (node->isLeaf) {
            return node->prediction;
        }
        
        bool goLeft = false;
        if (node->splitFeature == "ethnicity") {
            goLeft = (p.ethnicity == node->splitValueStr);
        } else if (node->splitFeature == "gender") {
            goLeft = (p.gender == node->splitValueStr);
        } else { // icd9
            goLeft = (p.icd9Code1 == node->splitValueInt);
        }
        
        if (goLeft) {
            return predictOne(node->left, p);
        } else {
            return predictOne(node->right, p);
        }
    }
    
public:
    DecisionTree(int maxD = 10, int minSamples = 2) 
        : root(nullptr), maxDepth(maxD), minSamplesSplit(minSamples) {}
    
    ~DecisionTree() {
        delete root;
    }
    
    void train(const vector<Patient>& data) {
        delete root;
        root = buildTree(data, 0);
    }
    
    int predict(const Patient& p) {
        if (!root) return 0;
        return predictOne(root, p);
    }
    
    double calculateAccuracy(const vector<Patient>& data) {
        if (data.empty()) return 0.0;
        
        int correct = 0;
        for (const auto& p : data) {
            int pred = predict(p);
            if (pred == p.expireFlag) correct++;
        }
        return (double)correct / data.size();
    }
    
    double calculateDeathRate(const vector<Patient>& data) {
        if (data.empty()) return 0.0;
        
        int totalPred = 0;
        for (const auto& p : data) {
            totalPred += predict(p);
        }
        return (double)totalPred / data.size();
    }
};

// Helper function to safely convert string to int
int safeStoi(const string& str) {
    if (str.empty()) return 0;
    try {
        // Remove spaces
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
            
            // Remove quotes if present
            auto removeQuotes = [](string& s) {
                s.erase(remove(s.begin(), s.end(), '\"'), s.end());
            };
            
            // SUBJECT_ID
            if (!getline(ss, temp, ',')) continue;
            removeQuotes(temp);
            p.subjectId = safeStoi(temp);
            
            // HADM_ID (skip)
            if (!getline(ss, temp, ',')) continue;
            
            // HOSPITAL_EXPIRE_FLAG
            if (!getline(ss, temp, ',')) continue;
            removeQuotes(temp);
            p.expireFlag = safeStoi(temp);
            
            // ADMITTIME (skip)
            if (!getline(ss, temp, ',')) continue;
            
            // ETHNICITY
            if (!getline(ss, p.ethnicity, ',')) continue;
            removeQuotes(p.ethnicity);
            // Trim spaces
            p.ethnicity.erase(0, p.ethnicity.find_first_not_of(" \t\r\n"));
            p.ethnicity.erase(p.ethnicity.find_last_not_of(" \t\r\n") + 1);
            
            // GENDER
            if (!getline(ss, p.gender, ',')) continue;
            removeQuotes(p.gender);
            // Trim spaces
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
                removeQuotes(temp);
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
    
    auto startTotal = chrono::high_resolution_clock::now();
    
    // Load data
    auto startLoad = chrono::high_resolution_clock::now();
    vector<Patient> data = loadData(argv[1]);
    auto endLoad = chrono::high_resolution_clock::now();
    double loadTime = chrono::duration<double>(endLoad - startLoad).count();
    
    cout << "Serial Implementation - Decision Tree" << endl;
    cout << "======================================" << endl;
    cout << "Data loaded: " << data.size() << " patients" << endl;
    cout << "Load time: " << loadTime << " seconds" << endl;
    
    // Split data (80% train, 20% test)
    int trainSize = data.size() * 0.8;
    vector<Patient> trainData(data.begin(), data.begin() + trainSize);
    vector<Patient> testData(data.begin() + trainSize, data.end());
    
    // Train model
    auto startTrain = chrono::high_resolution_clock::now();
    DecisionTree model(10, 5);  // max_depth=10, min_samples_split=5
    model.train(trainData);
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
    ofstream timingFile("timing_dt_serial.txt");
    timingFile << "load," << loadTime << endl;
    timingFile << "train," << trainTime << endl;
    timingFile << "eval," << evalTime << endl;
    timingFile << "total," << totalTime << endl;
    timingFile << "accuracy," << accuracy << endl;
    timingFile << "deathrate," << deathRate << endl;
    timingFile.close();
    
    return 0;
}
