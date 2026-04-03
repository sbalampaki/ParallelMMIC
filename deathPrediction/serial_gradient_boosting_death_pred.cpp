#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <limits>

using namespace std;

struct Patient {
    int subjectId;
    string ethnicity;
    string gender;
    int icd9Code1;
    int expireFlag;
};

// Regression tree node for gradient boosting
struct RegressionTreeNode {
    bool isLeaf;
    double leafValue;          // Mean residual for leaf nodes

    // Split info for internal nodes
    string splitFeature;       // "ethnicity", "gender", or "icd9"
    string splitValueStr;      // For categorical features
    int    splitValueInt;      // For ICD9 code

    RegressionTreeNode* left;
    RegressionTreeNode* right;

    RegressionTreeNode()
        : isLeaf(false), leafValue(0.0), splitValueInt(0),
          left(nullptr), right(nullptr) {}
    ~RegressionTreeNode() {
        delete left;
        delete right;
    }
};

// Shallow regression tree for boosting rounds
class RegressionTree {
private:
    RegressionTreeNode* root;
    int maxDepth;
    int minSamplesSplit;

    // Variance reduction (MSE-based) for a set of residuals
    double calculateVariance(const vector<double>& residuals) {
        if (residuals.empty()) return 0.0;
        double mean = 0.0;
        for (double r : residuals) mean += r;
        mean /= residuals.size();
        double var = 0.0;
        for (double r : residuals) var += (r - mean) * (r - mean);
        return var;
    }

    // Weighted variance reduction after a split
    double varianceReduction(const vector<double>& parent,
                             const vector<double>& left,
                             const vector<double>& right) {
        if (left.empty() || right.empty()) return -1e18;
        double n  = parent.size();
        double nl = left.size();
        double nr = right.size();
        return calculateVariance(parent)
             - (nl / n) * calculateVariance(left)
             - (nr / n) * calculateVariance(right);
    }

    void findBestSplit(const vector<Patient>& data,
                       const vector<double>& residuals,
                       string& bestFeature,
                       string& bestValueStr,
                       int&    bestValueInt,
                       double& bestGain) {
        bestGain    = -1e18;
        bestFeature = "";

        // Split on ethnicity
        map<string, int> ethSet;
        for (const auto& p : data) ethSet[p.ethnicity] = 1;
        for (const auto& kv : ethSet) {
            vector<bool> goLeft(data.size());
            for (size_t i = 0; i < data.size(); i++)
                goLeft[i] = (data[i].ethnicity == kv.first);
            vector<double> lRes, rRes;
            for (size_t i = 0; i < data.size(); i++) {
                if (goLeft[i]) lRes.push_back(residuals[i]);
                else           rRes.push_back(residuals[i]);
            }
            double gain = varianceReduction(residuals, lRes, rRes);
            if (gain > bestGain) {
                bestGain = gain;
                bestFeature = "ethnicity";
                bestValueStr = kv.first;
            }
        }

        // Split on gender
        map<string, int> genSet;
        for (const auto& p : data) genSet[p.gender] = 1;
        for (const auto& kv : genSet) {
            vector<bool> goLeft(data.size());
            for (size_t i = 0; i < data.size(); i++)
                goLeft[i] = (data[i].gender == kv.first);
            vector<double> lRes, rRes;
            for (size_t i = 0; i < data.size(); i++) {
                if (goLeft[i]) lRes.push_back(residuals[i]);
                else           rRes.push_back(residuals[i]);
            }
            double gain = varianceReduction(residuals, lRes, rRes);
            if (gain > bestGain) {
                bestGain = gain;
                bestFeature = "gender";
                bestValueStr = kv.first;
            }
        }

        // Split on ICD9 code (threshold split: icd9 <= value)
        map<int, int> icdSet;
        for (const auto& p : data) icdSet[p.icd9Code1] = 1;
        for (const auto& kv : icdSet) {
            vector<bool> goLeft(data.size());
            for (size_t i = 0; i < data.size(); i++)
                goLeft[i] = (data[i].icd9Code1 <= kv.first);
            vector<double> lRes, rRes;
            for (size_t i = 0; i < data.size(); i++) {
                if (goLeft[i]) lRes.push_back(residuals[i]);
                else           rRes.push_back(residuals[i]);
            }
            double gain = varianceReduction(residuals, lRes, rRes);
            if (gain > bestGain) {
                bestGain = gain;
                bestFeature = "icd9";
                bestValueInt = kv.first;
            }
        }
    }

    RegressionTreeNode* buildTree(const vector<Patient>& data,
                                  const vector<double>& residuals,
                                  int depth) {
        RegressionTreeNode* node = new RegressionTreeNode();

        if (data.empty()) {
            node->isLeaf = true;
            node->leafValue = 0.0;
            return node;
        }

        // Compute leaf value = mean residual (Newton step numerator / denominator)
        double mean = 0.0;
        for (double r : residuals) mean += r;
        mean /= residuals.size();

        if (depth >= maxDepth || (int)data.size() < minSamplesSplit) {
            node->isLeaf = true;
            node->leafValue = mean;
            return node;
        }

        string bestFeature, bestValueStr;
        int    bestValueInt = 0;
        double bestGain     = -1e18;

        findBestSplit(data, residuals, bestFeature, bestValueStr, bestValueInt, bestGain);

        if (bestGain <= 0.0 || bestFeature.empty()) {
            node->isLeaf = true;
            node->leafValue = mean;
            return node;
        }

        // Partition
        vector<Patient> lData, rData;
        vector<double>  lRes,  rRes;

        for (size_t i = 0; i < data.size(); i++) {
            bool goLeft = false;
            if (bestFeature == "ethnicity")
                goLeft = (data[i].ethnicity == bestValueStr);
            else if (bestFeature == "gender")
                goLeft = (data[i].gender == bestValueStr);
            else
                goLeft = (data[i].icd9Code1 <= bestValueInt);

            if (goLeft) { lData.push_back(data[i]); lRes.push_back(residuals[i]); }
            else         { rData.push_back(data[i]); rRes.push_back(residuals[i]); }
        }

        if (lData.empty() || rData.empty()) {
            node->isLeaf = true;
            node->leafValue = mean;
            return node;
        }

        node->splitFeature  = bestFeature;
        node->splitValueStr = bestValueStr;
        node->splitValueInt = bestValueInt;
        node->left  = buildTree(lData, lRes, depth + 1);
        node->right = buildTree(rData, rRes, depth + 1);

        return node;
    }

    double predictOne(RegressionTreeNode* node, const Patient& p) const {
        if (node->isLeaf) return node->leafValue;

        bool goLeft = false;
        if (node->splitFeature == "ethnicity")
            goLeft = (p.ethnicity == node->splitValueStr);
        else if (node->splitFeature == "gender")
            goLeft = (p.gender == node->splitValueStr);
        else
            goLeft = (p.icd9Code1 <= node->splitValueInt);

        return goLeft ? predictOne(node->left, p) : predictOne(node->right, p);
    }

public:
    RegressionTree(int maxD = 3, int minSamples = 5)
        : root(nullptr), maxDepth(maxD), minSamplesSplit(minSamples) {}

    ~RegressionTree() { delete root; }

    void train(const vector<Patient>& data, const vector<double>& residuals) {
        delete root;
        root = buildTree(data, residuals, 0);
    }

    double predict(const Patient& p) const {
        if (!root) return 0.0;
        return predictOne(root, p);
    }
};

// Gradient Boosting Classifier using logistic loss
class GradientBoostingClassifier {
private:
    vector<RegressionTree*> trees;
    double  F0;           // Initial log-odds
    double  learningRate;
    int     numTrees;
    int     maxDepth;

    static double sigmoid(double z) {
        return 1.0 / (1.0 + exp(-z));
    }

public:
    GradientBoostingClassifier(int nTrees = 100, double lr = 0.1, int depth = 3)
        : F0(0.0), learningRate(lr), numTrees(nTrees), maxDepth(depth) {}

    ~GradientBoostingClassifier() {
        for (auto* t : trees) delete t;
    }

    void train(const vector<Patient>& data) {
        int n = data.size();

        // Step 1: initialize with log-odds of the mean label
        double positiveCount = 0.0;
        for (const auto& p : data) positiveCount += p.expireFlag;
        double meanLabel = positiveCount / n;
        // Clamp to avoid log(0)
        meanLabel = max(1e-7, min(1.0 - 1e-7, meanLabel));
        F0 = log(meanLabel / (1.0 - meanLabel));

        // Running predictions (log-odds space)
        vector<double> F(n, F0);

        for (int t = 0; t < numTrees; t++) {
            // Step 2: compute pseudo-residuals = y - sigmoid(F)
            vector<double> residuals(n);
            for (int i = 0; i < n; i++) {
                double prob = sigmoid(F[i]);
                residuals[i] = data[i].expireFlag - prob;
            }

            // Step 3: fit a regression tree to pseudo-residuals
            RegressionTree* tree = new RegressionTree(maxDepth, 5);
            tree->train(data, residuals);
            trees.push_back(tree);

            // Step 4: update predictions
            for (int i = 0; i < n; i++) {
                F[i] += learningRate * tree->predict(data[i]);
            }
        }
    }

    // Returns probability of death
    double predictProba(const Patient& p) const {
        double score = F0;
        for (const auto* t : trees)
            score += learningRate * t->predict(p);
        return sigmoid(score);
    }

    int predict(const Patient& p) const {
        return predictProba(p) >= 0.5 ? 1 : 0;
    }

    double calculateAccuracy(const vector<Patient>& data) const {
        int correct = 0;
        for (const auto& p : data) {
            if (predict(p) == p.expireFlag) correct++;
        }
        return (double)correct / data.size();
    }

    double calculateDeathRate(const vector<Patient>& data) const {
        double total = 0.0;
        for (const auto& p : data) total += predictProba(p);
        return total / data.size();
    }
};

// ─── Data loading ────────────────────────────────────────────────────────────

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

    int lineNum = 1;
    while (getline(file, line)) {
        lineNum++;
        if (line.empty()) continue;

        try {
            stringstream ss(line);
            Patient p;
            string temp;

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
            p.ethnicity.erase(0, p.ethnicity.find_first_not_of(" \t\r\n"));
            p.ethnicity.erase(p.ethnicity.find_last_not_of(" \t\r\n") + 1);

            // GENDER
            if (!getline(ss, p.gender, ',')) continue;
            removeQuotes(p.gender);
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

// ─── Main ────────────────────────────────────────────────────────────────────

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

    cout << "Serial Implementation - Gradient Boosting" << endl;
    cout << "==========================================" << endl;
    cout << "Data loaded: " << data.size() << " patients" << endl;
    cout << "Load time: " << loadTime << " seconds" << endl;

    // Split data (80% train, 20% test)
    int trainSize = (int)(data.size() * 0.8);
    vector<Patient> trainData(data.begin(), data.begin() + trainSize);
    vector<Patient> testData(data.begin() + trainSize, data.end());

    cout << "Training samples: " << trainData.size()
         << "  Test samples: " << testData.size() << endl;
    cout << "Model: Gradient Boosting (100 trees, depth=3, lr=0.1)" << endl;

    // Train model
    auto startTrain = chrono::high_resolution_clock::now();
    GradientBoostingClassifier model(100, 0.1, 3);
    model.train(trainData);
    auto endTrain = chrono::high_resolution_clock::now();
    double trainTime = chrono::duration<double>(endTrain - startTrain).count();

    cout << "Training time: " << trainTime << " seconds" << endl;

    // Evaluate
    auto startEval = chrono::high_resolution_clock::now();
    double accuracy  = model.calculateAccuracy(testData);
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
    ofstream timingFile("timing_gb_serial.txt");
    timingFile << "load,"      << loadTime  << endl;
    timingFile << "train,"     << trainTime << endl;
    timingFile << "eval,"      << evalTime  << endl;
    timingFile << "total,"     << totalTime << endl;
    timingFile << "accuracy,"  << accuracy  << endl;
    timingFile << "deathrate," << deathRate << endl;
    timingFile.close();

    return 0;
}
