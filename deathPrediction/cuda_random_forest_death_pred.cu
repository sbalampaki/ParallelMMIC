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

// Flat (GPU-ready) tree node – stored in a contiguous array per tree
struct FlatNode {
    int feature;      // 0=ethnicity, 1=gender, 2=icd9, -1=leaf
    int strIdx;       // index into unique-value array for categorical features
    int intVal;       // ICD9 exact-match value
    int leftIdx;      // index of left child in the per-tree flat array (-1 if none)
    int rightIdx;     // index of right child
    int label;        // majority-class prediction (valid only at leaves)
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

    // Serialise tree into a flat array (DFS order, indices pre-assigned)
    void serialize(vector<FlatNode>& flat,
                   const map<string, int>& ethMap,
                   const map<string, int>& genMap) const {
        if (!root) return;

        // BFS-style index assignment then DFS fill
        struct Frame { TreeNode* node; int idx; };
        vector<Frame> stack;
        // Pre-count nodes to reserve
        function<int(TreeNode*)> countNodes = [&](TreeNode* n) -> int {
            if (!n) return 0;
            return 1 + countNodes(n->left) + countNodes(n->right);
        };
        int totalNodes = countNodes(root);
        flat.resize(totalNodes);

        // DFS with explicit index assignment
        int nextIdx = 0;
        function<void(TreeNode*, int)> fillFlat = [&](TreeNode* n, int idx) {
            if (n->isLeaf) {
                flat[idx].feature  = -1;
                flat[idx].strIdx   = -1;
                flat[idx].intVal   = 0;
                flat[idx].leftIdx  = -1;
                flat[idx].rightIdx = -1;
                flat[idx].label    = n->prediction;
                return;
            }

            int featureCode = 0;
            int strIdxVal   = -1;
            if (n->splitFeature == "gender") {
                featureCode = 1;
                auto it = genMap.find(n->splitValueStr);
                if (it != genMap.end()) strIdxVal = it->second;
            } else if (n->splitFeature == "ethnicity") {
                featureCode = 0;
                auto it = ethMap.find(n->splitValueStr);
                if (it != ethMap.end()) strIdxVal = it->second;
            } else {
                featureCode = 2;
            }

            int lIdx = ++nextIdx;
            int rIdx = ++nextIdx;

            flat[idx].feature  = featureCode;
            flat[idx].strIdx   = strIdxVal;
            flat[idx].intVal   = n->splitValueInt;
            flat[idx].leftIdx  = lIdx;
            flat[idx].rightIdx = rIdx;
            flat[idx].label    = -1;

            fillFlat(n->left,  lIdx);
            fillFlat(n->right, rIdx);
        };

        fillFlat(root, nextIdx);
    }
};

// ─── CUDA kernel ──────────────────────────────────────────────────────────────
//
// Each thread handles one test sample.  It traverses every tree in the forest
// and accumulates votes, then stores the final vote count.
// Trees are stored in a flattened layout: tree t starts at offset t * MAX_NODES.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void rfPredictKernel(
    const int* ethIdx,       // [n] patient ethnicity index
    const int* genIdx,       // [n] patient gender index
    const int* icd9Val,      // [n] patient ICD9 code
    const int* nodeFeature,  // [numTrees * maxNodes]
    const int* nodeStrIdx,   // [numTrees * maxNodes]
    const int* nodeIntVal,   // [numTrees * maxNodes]
    const int* nodeLeft,     // [numTrees * maxNodes]
    const int* nodeRight,    // [numTrees * maxNodes]
    const int* nodeLabel,    // [numTrees * maxNodes]
    int* votes,              // [n] output vote counts
    int n,
    int numTrees,
    int maxNodes
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int patEth = ethIdx[i];
    int patGen = genIdx[i];
    int patIcd = icd9Val[i];

    int voteCount = 0;
    for (int t = 0; t < numTrees; t++) {
        int base = t * maxNodes;
        int cur  = base;  // root is always index 0 within a tree's segment

        // Traverse tree until leaf
        while (nodeFeature[cur] != -1) {
            int feat = nodeFeature[cur];
            bool goLeft = false;
            if (feat == 0) {          // ethnicity exact match
                goLeft = (patEth == nodeStrIdx[cur]);
            } else if (feat == 1) {   // gender exact match
                goLeft = (patGen == nodeStrIdx[cur]);
            } else {                  // icd9 exact match
                goLeft = (patIcd == nodeIntVal[cur]);
            }
            int next = goLeft ? nodeLeft[cur] : nodeRight[cur];
            if (next < 0) break;  // safety guard
            cur = base + next;
        }
        voteCount += nodeLabel[cur];
    }
    votes[i] = voteCount;
}

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
    cout << "CUDA Implementation - Random Forest" << endl;
    cout << "====================================" << endl;
    cout << "Using GPU: " << prop.name << endl;
    cout << "Compute Capability: " << prop.major << "." << prop.minor << endl;

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
    cout << "Model: Random Forest (50 trees, CPU training, GPU-accelerated voting)" << endl;

    // ── Build vocabulary maps for categorical features ─────────────────────
    map<string, int> ethMap, genMap;
    int ethIdx = 0, genIdx = 0;
    for (const auto& p : data) {
        if (ethMap.find(p.ethnicity) == ethMap.end()) ethMap[p.ethnicity] = ethIdx++;
        if (genMap.find(p.gender)    == genMap.end()) genMap[p.gender]    = genIdx++;
    }

    // ── Train all trees on CPU ─────────────────────────────────────────────
    const int NUM_TREES  = 50;
    const int MAX_DEPTH  = 10;
    const int MIN_SAMP   = 5;

    auto startTrain = chrono::high_resolution_clock::now();

    int n = (int)trainData.size();
    vector<DecisionTree*> trees(NUM_TREES, nullptr);
    for (int t = 0; t < NUM_TREES; t++) {
        mt19937 treeRng(42u + (unsigned int)t * 1000003u);
        uniform_int_distribution<int> dist(0, n - 1);
        vector<Patient> bootstrap;
        bootstrap.reserve(n);
        for (int i = 0; i < n; i++) bootstrap.push_back(trainData[dist(treeRng)]);
        trees[t] = new DecisionTree(MAX_DEPTH, MIN_SAMP);
        trees[t]->train(bootstrap, &treeRng);
    }

    // ── Serialise trees to flat arrays ────────────────────────────────────
    int MAX_NODES = 0;
    vector<vector<FlatNode>> flatTrees(NUM_TREES);
    for (int t = 0; t < NUM_TREES; t++) {
        trees[t]->serialize(flatTrees[t], ethMap, genMap);
        MAX_NODES = max(MAX_NODES, (int)flatTrees[t].size());
    }
    MAX_NODES = max(MAX_NODES, 1);  // guard against empty tree

    // Pack into 1D arrays padded to MAX_NODES
    int total = NUM_TREES * MAX_NODES;
    vector<int> h_nodeFeature(total, -1);
    vector<int> h_nodeStrIdx(total, -1);
    vector<int> h_nodeIntVal(total, 0);
    vector<int> h_nodeLeft(total, -1);
    vector<int> h_nodeRight(total, -1);
    vector<int> h_nodeLabel(total, 0);

    for (int t = 0; t < NUM_TREES; t++) {
        int base = t * MAX_NODES;
        for (int i = 0; i < (int)flatTrees[t].size(); i++) {
            const FlatNode& fn = flatTrees[t][i];
            h_nodeFeature[base + i] = fn.feature;
            h_nodeStrIdx[base + i]  = fn.strIdx;
            h_nodeIntVal[base + i]  = fn.intVal;
            h_nodeLeft[base + i]    = fn.leftIdx;
            h_nodeRight[base + i]   = fn.rightIdx;
            h_nodeLabel[base + i]   = fn.label;
        }
        // Remaining slots are already initialised to leaf-like values (feature=-1, label=0)
    }

    auto endTrain = chrono::high_resolution_clock::now();
    double trainTime = chrono::duration<double>(endTrain - startTrain).count();
    cout << "Training time (CPU trees + serialisation): " << trainTime << " seconds" << endl;

    // ── Encode test patients ───────────────────────────────────────────────
    int nTest = (int)testData.size();
    vector<int> h_ethIdx(nTest), h_genIdx(nTest), h_icd9Val(nTest), h_labels(nTest);
    for (int i = 0; i < nTest; i++) {
        const auto& p = testData[i];
        h_ethIdx[i]  = ethMap.count(p.ethnicity) ? ethMap.at(p.ethnicity) : -1;
        h_genIdx[i]  = genMap.count(p.gender)    ? genMap.at(p.gender)    : -1;
        h_icd9Val[i] = p.icd9Code1;
        h_labels[i]  = p.expireFlag;
    }

    auto startEval = chrono::high_resolution_clock::now();

    // ── GPU allocation & transfer ──────────────────────────────────────────
    int *d_ethIdx, *d_genIdx, *d_icd9Val;
    int *d_nodeFeature, *d_nodeStrIdx, *d_nodeIntVal;
    int *d_nodeLeft, *d_nodeRight, *d_nodeLabel;
    int *d_votes;

    CUDA_CHECK(cudaMalloc(&d_ethIdx,       nTest  * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_genIdx,       nTest  * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_icd9Val,      nTest  * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nodeFeature,  total  * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nodeStrIdx,   total  * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nodeIntVal,   total  * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nodeLeft,     total  * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nodeRight,    total  * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nodeLabel,    total  * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_votes,        nTest  * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_ethIdx,      h_ethIdx.data(),       nTest * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_genIdx,      h_genIdx.data(),       nTest * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_icd9Val,     h_icd9Val.data(),      nTest * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nodeFeature, h_nodeFeature.data(),  total * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nodeStrIdx,  h_nodeStrIdx.data(),   total * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nodeIntVal,  h_nodeIntVal.data(),   total * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nodeLeft,    h_nodeLeft.data(),     total * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nodeRight,   h_nodeRight.data(),    total * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nodeLabel,   h_nodeLabel.data(),    total * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_votes, 0, nTest * sizeof(int)));

    int blockSize = 256;
    int numBlocks = (nTest + blockSize - 1) / blockSize;

    rfPredictKernel<<<numBlocks, blockSize>>>(
        d_ethIdx, d_genIdx, d_icd9Val,
        d_nodeFeature, d_nodeStrIdx, d_nodeIntVal,
        d_nodeLeft, d_nodeRight, d_nodeLabel,
        d_votes, nTest, NUM_TREES, MAX_NODES
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    vector<int> h_votes(nTest);
    CUDA_CHECK(cudaMemcpy(h_votes.data(), d_votes, nTest * sizeof(int), cudaMemcpyDeviceToHost));

    // ── Compute metrics ────────────────────────────────────────────────────
    int correct = 0;
    double totalProba = 0.0;
    for (int i = 0; i < nTest; i++) {
        int pred = (h_votes[i] * 2 >= NUM_TREES) ? 1 : 0;
        if (pred == h_labels[i]) correct++;
        totalProba += (double)h_votes[i] / NUM_TREES;
    }
    double accuracy  = (double)correct / nTest;
    double deathRate = totalProba / nTest;

    auto endEval = chrono::high_resolution_clock::now();
    double evalTime = chrono::duration<double>(endEval - startEval).count();

    auto endTotal = chrono::high_resolution_clock::now();
    double totalTime = chrono::duration<double>(endTotal - startTotal).count();

    cout << "Evaluation time (GPU voting): " << evalTime << " seconds" << endl;
    cout << "Total execution time: " << totalTime << " seconds" << endl;
    cout << "\nResults:" << endl;
    cout << "Accuracy: " << (accuracy * 100) << "%" << endl;
    cout << "Predicted Death Rate: " << (deathRate * 100) << "%" << endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_ethIdx));
    CUDA_CHECK(cudaFree(d_genIdx));
    CUDA_CHECK(cudaFree(d_icd9Val));
    CUDA_CHECK(cudaFree(d_nodeFeature));
    CUDA_CHECK(cudaFree(d_nodeStrIdx));
    CUDA_CHECK(cudaFree(d_nodeIntVal));
    CUDA_CHECK(cudaFree(d_nodeLeft));
    CUDA_CHECK(cudaFree(d_nodeRight));
    CUDA_CHECK(cudaFree(d_nodeLabel));
    CUDA_CHECK(cudaFree(d_votes));
    for (auto* t : trees) delete t;

    ofstream timingFile("timing_rf_cuda.txt");
    timingFile << "load,"      << loadTime  << endl;
    timingFile << "train,"     << trainTime << endl;
    timingFile << "eval,"      << evalTime  << endl;
    timingFile << "total,"     << totalTime << endl;
    timingFile << "accuracy,"  << accuracy  << endl;
    timingFile << "deathrate," << deathRate << endl;
    timingFile.close();

    return 0;
}
