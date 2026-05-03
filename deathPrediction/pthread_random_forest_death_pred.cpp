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
#include <random>
#include <pthread.h>

using namespace std;

struct Patient {
    int subjectId;
    string ethnicity;
    string gender;
    int icd9Code1;
    int expireFlag;
};

struct TreeNode {
    bool isLeaf;
    int prediction;

    string splitFeature;
    string splitValueStr;
    int splitValueInt;

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
    mt19937* rng;

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

    double calculateInformationGain(const vector<Patient>& data,
                                     const vector<Patient>& left,
                                     const vector<Patient>& right) {
        double parentEntropy = calculateEntropy(data);
        double leftWeight  = (double)left.size()  / data.size();
        double rightWeight = (double)right.size() / data.size();
        return parentEntropy - (leftWeight  * calculateEntropy(left) +
                                rightWeight * calculateEntropy(right));
    }

    vector<string> sampleFeatures() {
        vector<string> all = {"ethnicity", "gender", "icd9"};
        if (rng == nullptr) return all;
        shuffle(all.begin(), all.end(), *rng);
        return vector<string>(all.begin(), all.begin() + 2);
    }

    void findBestSplit(const vector<Patient>& data,
                       string& bestFeature,
                       string& bestValueStr,
                       int& bestValueInt,
                       double& bestGain) {
        bestGain    = 0.0;
        bestFeature = "";

        vector<string> featuresToTry = sampleFeatures();

        for (const auto& feat : featuresToTry) {
            if (feat == "ethnicity") {
                map<string, int> groups;
                for (const auto& p : data) groups[p.ethnicity]++;
                for (const auto& kv : groups) {
                    vector<Patient> left, right;
                    for (const auto& p : data) {
                        if (p.ethnicity == kv.first) left.push_back(p);
                        else                          right.push_back(p);
                    }
                    if (!left.empty() && !right.empty()) {
                        double gain = calculateInformationGain(data, left, right);
                        if (gain > bestGain) {
                            bestGain     = gain;
                            bestFeature  = "ethnicity";
                            bestValueStr = kv.first;
                        }
                    }
                }
            } else if (feat == "gender") {
                map<string, int> groups;
                for (const auto& p : data) groups[p.gender]++;
                for (const auto& kv : groups) {
                    vector<Patient> left, right;
                    for (const auto& p : data) {
                        if (p.gender == kv.first) left.push_back(p);
                        else                       right.push_back(p);
                    }
                    if (!left.empty() && !right.empty()) {
                        double gain = calculateInformationGain(data, left, right);
                        if (gain > bestGain) {
                            bestGain     = gain;
                            bestFeature  = "gender";
                            bestValueStr = kv.first;
                        }
                    }
                }
            } else {
                map<int, int> groups;
                for (const auto& p : data) groups[p.icd9Code1]++;
                for (const auto& kv : groups) {
                    vector<Patient> left, right;
                    for (const auto& p : data) {
                        if (p.icd9Code1 == kv.first) left.push_back(p);
                        else                          right.push_back(p);
                    }
                    if (!left.empty() && !right.empty()) {
                        double gain = calculateInformationGain(data, left, right);
                        if (gain > bestGain) {
                            bestGain     = gain;
                            bestFeature  = "icd9";
                            bestValueInt = kv.first;
                        }
                    }
                }
            }
        }
    }

    TreeNode* buildTree(const vector<Patient>& data, int depth) {
        TreeNode* node = new TreeNode();

        if (data.empty()) {
            node->isLeaf     = true;
            node->prediction = 0;
            return node;
        }

        int positive = 0;
        for (const auto& p : data) {
            if (p.expireFlag == 1) positive++;
        }

        if (positive == 0 || positive == (int)data.size() ||
            depth >= maxDepth || (int)data.size() < minSamplesSplit) {
            node->isLeaf     = true;
            node->prediction = (positive > (int)data.size() / 2) ? 1 : 0;
            return node;
        }

        string bestFeature, bestValueStr;
        int    bestValueInt = 0;
        double bestGain     = 0.0;
        findBestSplit(data, bestFeature, bestValueStr, bestValueInt, bestGain);

        if (bestGain <= 0.0 || bestFeature.empty()) {
            node->isLeaf     = true;
            node->prediction = (positive > (int)data.size() / 2) ? 1 : 0;
            return node;
        }

        vector<Patient> left, right;
        if (bestFeature == "ethnicity") {
            for (const auto& p : data) {
                if (p.ethnicity == bestValueStr) left.push_back(p);
                else                              right.push_back(p);
            }
            node->splitFeature  = "ethnicity";
            node->splitValueStr = bestValueStr;
        } else if (bestFeature == "gender") {
            for (const auto& p : data) {
                if (p.gender == bestValueStr) left.push_back(p);
                else                           right.push_back(p);
            }
            node->splitFeature  = "gender";
            node->splitValueStr = bestValueStr;
        } else {
            for (const auto& p : data) {
                if (p.icd9Code1 == bestValueInt) left.push_back(p);
                else                              right.push_back(p);
            }
            node->splitFeature  = "icd9";
            node->splitValueInt = bestValueInt;
        }

        node->left  = buildTree(left,  depth + 1);
        node->right = buildTree(right, depth + 1);
        return node;
    }

    int predictOne(TreeNode* node, const Patient& p) const {
        if (node->isLeaf) return node->prediction;
        bool goLeft = false;
        if (node->splitFeature == "ethnicity") goLeft = (p.ethnicity == node->splitValueStr);
        else if (node->splitFeature == "gender") goLeft = (p.gender == node->splitValueStr);
        else                                     goLeft = (p.icd9Code1 == node->splitValueInt);
        return goLeft ? predictOne(node->left, p) : predictOne(node->right, p);
    }

public:
    DecisionTree(int maxD = 10, int minSamples = 5)
        : root(nullptr), maxDepth(maxD), minSamplesSplit(minSamples), rng(nullptr) {}

    ~DecisionTree() { delete root; }

    void train(const vector<Patient>& data, mt19937* featureRng = nullptr) {
        rng = featureRng;
        delete root;
        root = buildTree(data, 0);
        rng = nullptr;
    }

    int predict(const Patient& p) const {
        if (!root) return 0;
        return predictOne(root, p);
    }
};

// ─── Pthread thread data structures ───────────────────────────────────────────

struct TrainThreadData {
    const vector<Patient>* trainData;
    vector<DecisionTree*>* trees;  // shared pre-allocated vector
    int treeStart;
    int treeEnd;
    int maxDepth;
    int minSamplesSplit;
    unsigned int seed;
};

struct PredictThreadData {
    const vector<Patient>* testData;
    const vector<DecisionTree*>* trees;
    int start;
    int end;
    int correct;
    double totalProba;
    int numTrees;
};

void* trainTreeRange(void* arg) {
    TrainThreadData* td = (TrainThreadData*)arg;
    int n = (int)td->trainData->size();

    for (int t = td->treeStart; t < td->treeEnd; t++) {
        mt19937 treeRng(td->seed + (unsigned int)t * 1000003u);
        uniform_int_distribution<int> dist(0, n - 1);

        vector<Patient> bootstrap;
        bootstrap.reserve(n);
        for (int i = 0; i < n; i++) {
            bootstrap.push_back((*td->trainData)[dist(treeRng)]);
        }

        DecisionTree* tree = new DecisionTree(td->maxDepth, td->minSamplesSplit);
        tree->train(bootstrap, &treeRng);
        (*td->trees)[t] = tree;  // no race condition: each thread writes to disjoint indices
    }
    return NULL;
}

void* predictRange(void* arg) {
    PredictThreadData* pd = (PredictThreadData*)arg;
    pd->correct    = 0;
    pd->totalProba = 0.0;

    for (int i = pd->start; i < pd->end; i++) {
        const Patient& p = (*pd->testData)[i];
        int votes = 0;
        for (const auto* t : *pd->trees) votes += t->predict(p);
        int pred = (votes * 2 >= pd->numTrees) ? 1 : 0;
        if (pred == p.expireFlag) pd->correct++;
        pd->totalProba += (double)votes / pd->numTrees;
    }
    return NULL;
}

// ─── Pthread Random Forest ────────────────────────────────────────────────────
//
// Parallelism strategy:
//   Training: spawn numThreads Pthreads, each building a contiguous range of
//             trees from the ensemble.  Trees are written to pre-allocated
//             index positions so no mutex is required.
//   Prediction: spawn numThreads Pthreads, each evaluating a contiguous slice
//               of the test set.
// ─────────────────────────────────────────────────────────────────────────────
class RandomForestClassifierPThread {
private:
    vector<DecisionTree*> trees;
    int numTrees;
    int maxDepth;
    int minSamplesSplit;
    int numThreads;

public:
    RandomForestClassifierPThread(int nTrees = 50, int maxD = 10, int minSamples = 5, int nThreads = 4)
        : numTrees(nTrees), maxDepth(maxD), minSamplesSplit(minSamples), numThreads(nThreads) {
        trees.resize(numTrees, nullptr);
    }

    ~RandomForestClassifierPThread() {
        for (auto* t : trees) delete t;
    }

    void train(const vector<Patient>& data, unsigned int seed = 42) {
        vector<pthread_t> pthreads(numThreads);
        vector<TrainThreadData> tdArray(numThreads);

        int treesPerThread = numTrees / numThreads;

        for (int t = 0; t < numThreads; t++) {
            tdArray[t].trainData       = &data;
            tdArray[t].trees           = &trees;
            tdArray[t].treeStart       = t * treesPerThread;
            tdArray[t].treeEnd         = (t == numThreads - 1) ? numTrees : (t + 1) * treesPerThread;
            tdArray[t].maxDepth        = maxDepth;
            tdArray[t].minSamplesSplit = minSamplesSplit;
            tdArray[t].seed            = seed;

            pthread_create(&pthreads[t], NULL, trainTreeRange, &tdArray[t]);
        }

        for (int t = 0; t < numThreads; t++) {
            pthread_join(pthreads[t], NULL);
        }
    }

    double calculateAccuracy(const vector<Patient>& testData) {
        int n = (int)testData.size();
        vector<pthread_t> pthreads(numThreads);
        vector<PredictThreadData> pdArray(numThreads);

        int chunkSize = n / numThreads;
        for (int t = 0; t < numThreads; t++) {
            pdArray[t].testData   = &testData;
            pdArray[t].trees      = &trees;
            pdArray[t].start      = t * chunkSize;
            pdArray[t].end        = (t == numThreads - 1) ? n : (t + 1) * chunkSize;
            pdArray[t].numTrees   = numTrees;
            pthread_create(&pthreads[t], NULL, predictRange, &pdArray[t]);
        }

        int totalCorrect = 0;
        for (int t = 0; t < numThreads; t++) {
            pthread_join(pthreads[t], NULL);
            totalCorrect += pdArray[t].correct;
        }
        return (double)totalCorrect / n;
    }

    double calculateDeathRate(const vector<Patient>& testData) {
        int n = (int)testData.size();
        vector<pthread_t> pthreads(numThreads);
        vector<PredictThreadData> pdArray(numThreads);

        int chunkSize = n / numThreads;
        for (int t = 0; t < numThreads; t++) {
            pdArray[t].testData   = &testData;
            pdArray[t].trees      = &trees;
            pdArray[t].start      = t * chunkSize;
            pdArray[t].end        = (t == numThreads - 1) ? n : (t + 1) * chunkSize;
            pdArray[t].numTrees   = numTrees;
            pthread_create(&pthreads[t], NULL, predictRange, &pdArray[t]);
        }

        double totalProba = 0.0;
        for (int t = 0; t < numThreads; t++) {
            pthread_join(pthreads[t], NULL);
            totalProba += pdArray[t].totalProba;
        }
        return totalProba / n;
    }
};

// ─── Data loading ─────────────────────────────────────────────────────────────

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
    getline(file, line);

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

            if (!getline(ss, temp, ',')) continue;
            removeQuotes(temp);
            p.subjectId = safeStoi(temp);

            if (!getline(ss, temp, ',')) continue;

            if (!getline(ss, temp, ',')) continue;
            removeQuotes(temp);
            p.expireFlag = safeStoi(temp);

            if (!getline(ss, temp, ',')) continue;

            if (!getline(ss, p.ethnicity, ',')) continue;
            removeQuotes(p.ethnicity);
            p.ethnicity.erase(0, p.ethnicity.find_first_not_of(" \t\r\n"));
            p.ethnicity.erase(p.ethnicity.find_last_not_of(" \t\r\n") + 1);

            if (!getline(ss, p.gender, ',')) continue;
            removeQuotes(p.gender);
            p.gender.erase(0, p.gender.find_first_not_of(" \t\r\n"));
            p.gender.erase(p.gender.find_last_not_of(" \t\r\n") + 1);

            if (!getline(ss, temp, ',')) continue;
            if (!getline(ss, temp, ',')) continue;

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

// ─── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <data_file> [num_threads]" << endl;
        return 1;
    }

    int numThreads = (argc > 2) ? atoi(argv[2]) : 4;

    cout << "Pthreads Implementation - Random Forest" << endl;
    cout << "========================================" << endl;
    cout << "Number of threads: " << numThreads << endl;

    auto startTotal = chrono::high_resolution_clock::now();

    auto startLoad = chrono::high_resolution_clock::now();
    vector<Patient> data = loadData(argv[1]);
    auto endLoad = chrono::high_resolution_clock::now();
    double loadTime = chrono::duration<double>(endLoad - startLoad).count();

    cout << "Data loaded: " << data.size() << " patients" << endl;
    cout << "Load time: " << loadTime << " seconds" << endl;

    int trainSize = (int)(data.size() * 0.8);
    vector<Patient> trainData(data.begin(), data.begin() + trainSize);
    vector<Patient> testData(data.begin() + trainSize, data.end());

    cout << "Training samples: " << trainData.size()
         << "  Test samples: " << testData.size() << endl;
    cout << "Model: Random Forest (50 trees, max_depth=10, Pthreads parallel tree building)" << endl;

    auto startTrain = chrono::high_resolution_clock::now();
    RandomForestClassifierPThread model(50, 10, 5, numThreads);
    model.train(trainData, 42);
    auto endTrain = chrono::high_resolution_clock::now();
    double trainTime = chrono::duration<double>(endTrain - startTrain).count();

    cout << "Training time: " << trainTime << " seconds" << endl;

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

    ofstream timingFile("timing_rf_pthread.txt");
    timingFile << "load,"      << loadTime  << endl;
    timingFile << "train,"     << trainTime << endl;
    timingFile << "eval,"      << evalTime  << endl;
    timingFile << "total,"     << totalTime << endl;
    timingFile << "accuracy,"  << accuracy  << endl;
    timingFile << "deathrate," << deathRate << endl;
    timingFile << "threads,"   << numThreads << endl;
    timingFile.close();

    return 0;
}
