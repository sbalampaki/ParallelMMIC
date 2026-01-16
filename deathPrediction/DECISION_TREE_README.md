# Decision Tree Implementation

## Overview

In addition to Logistic Regression, this project now includes a **Decision Tree** classifier for death rate prediction. Decision Trees offer a complementary approach to Logistic Regression with different characteristics that make them suitable for different scenarios.

## Why Decision Trees?

Decision Trees are an excellent addition to this project because:

1. **Interpretability**: Decision trees create clear, human-readable decision rules
2. **Non-linear Patterns**: Unlike logistic regression, decision trees can capture complex, non-linear relationships
3. **No Feature Scaling**: Decision trees work directly with categorical features without normalization
4. **Complementary Approach**: Provides an alternative ML algorithm for comparison

## Implementation Details

### Algorithm
- **Splitting Criterion**: Information Gain (Entropy-based)
- **Features**: Ethnicity, Gender, ICD9 Code
- **Max Depth**: 10 levels (configurable)
- **Min Samples Split**: 5 samples (configurable)

### How It Works
1. **Training Phase**: Recursively splits data to maximize information gain
   - Evaluates all possible splits on each feature
   - Selects the split that best separates death vs survival cases
   - Continues until stopping criteria are met (max depth, min samples, or pure node)

2. **Prediction Phase**: Traverses the tree based on patient features
   - Follows decision rules at each node
   - Returns the majority class at the leaf node

## Building

Build the Decision Tree implementation:
```bash
cd deathPrediction
make dt_serial
```

## Running

Execute the Decision Tree classifier:
```bash
./serial_decision_tree_death_pred mimic_data.csv
```

## Example Output

```
Serial Implementation - Decision Tree
======================================
Data loaded: 10000 patients
Load time: 0.0072 seconds
Training time: 0.0721 seconds
Evaluation time: 0.0002 seconds
Total execution time: 0.0800 seconds

Results:
Accuracy: 49.55%
Predicted Death Rate: 19.15%
```

## Comparison with Logistic Regression

For a comprehensive comparison with actual execution results, see **[ALGORITHM_COMPARISON.md](ALGORITHM_COMPARISON.md)** which includes:
- Detailed performance metrics from actual runs
- Execution time analysis (Decision Tree is 4.4x faster overall)
- Accuracy comparison and analysis
- Use case recommendations

Quick comparison summary:

| Aspect | Logistic Regression | Decision Tree |
|--------|-------------------|---------------|
| **Model Type** | Linear | Non-linear |
| **Training** | Gradient descent | Recursive splitting |
| **Interpretability** | Coefficients | Decision rules |
| **Feature Handling** | Requires encoding | Direct categorical support |
| **Overfitting Risk** | Lower | Higher (if not pruned) |
| **Training Speed** | Moderate | Fast |
| **Prediction Speed** | Very fast | Fast |

## Performance Characteristics

### Training Time
Decision Trees typically have:
- **Faster training** on this dataset (~0.07s vs ~0.29s for Logistic Regression)
- O(n × m × log n) complexity where n = samples, m = features
- No iterative optimization required

### Accuracy
Both algorithms provide different insights:
- **Logistic Regression**: Better when relationships are linear
- **Decision Tree**: Better when relationships are complex or non-linear

### Scalability
The decision tree implementation can be parallelized using:
- Parallel feature evaluation during split selection
- Parallel prediction for multiple patients
- Distributed tree building (ensemble methods)

## Future Enhancements

Potential improvements for the Decision Tree implementation:

1. **Pruning**: Add post-pruning to reduce overfitting
2. **Parallel Training**: OpenMP parallel split evaluation
3. **Random Forest**: Ensemble of multiple decision trees
4. **Better Splitting**: Try Gini impurity as alternative to entropy
5. **Continuous Features**: Support for continuous ICD9 code ranges
6. **Cross-Validation**: K-fold validation for parameter tuning

## Use Cases

**Use Decision Tree when:**
- You need interpretable decision rules for clinical decisions
- The relationship between features and outcome is non-linear
- You want faster training times
- Feature interactions are important

**Use Logistic Regression when:**
- You need probabilistic outputs
- The relationship is approximately linear
- You want a simpler, more stable model
- You need to explain feature importance through coefficients

## Technical Notes

### Data Structure
```cpp
struct TreeNode {
    bool isLeaf;
    int prediction;              // For leaf nodes
    string splitFeature;         // "ethnicity", "gender", or "icd9"
    string splitValueStr;        // For categorical features
    int splitValueInt;           // For ICD9 code
    TreeNode* left;              // Patients matching the split
    TreeNode* right;             // Patients not matching the split
};
```

### Entropy Calculation
```
H(S) = -p₁ log₂(p₁) - p₀ log₂(p₀)
```
Where p₁ = probability of death, p₀ = probability of survival

### Information Gain
```
IG(S, A) = H(S) - Σ(|Sᵥ|/|S|) × H(Sᵥ)
```
Where A = splitting attribute, Sᵥ = subset for each value

## Files

- **serial_decision_tree_death_pred.cpp** - Complete Decision Tree implementation
- **timing_dt_serial.txt** - Performance timing output

## References

- [Decision Tree Learning (Wikipedia)](https://en.wikipedia.org/wiki/Decision_tree_learning)
- [Information Gain](https://en.wikipedia.org/wiki/Information_gain_in_decision_trees)
- [Entropy (Information Theory)](https://en.wikipedia.org/wiki/Entropy_(information_theory))
