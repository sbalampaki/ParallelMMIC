# Algorithm Comparison: Logistic Regression vs Decision Tree

## Executive Summary

This document presents a comprehensive comparison between the two machine learning algorithms implemented in this project for death rate prediction on the MIMIC clinical dataset:
- **Logistic Regression** (using gradient descent)
- **Decision Tree** (using entropy-based information gain)

Both algorithms have been executed on the same dataset (`mimic_data.csv` with 10,000 patients) and their results are compared below.

---

## Experimental Setup

- **Dataset**: `mimic_data.csv`
- **Number of Patients**: 10,000
- **Features**: Ethnicity, Gender, ICD9 Primary Diagnosis Code
- **Target**: Binary classification (Death vs Survival)
- **Test Environment**: Serial implementation (baseline, single-threaded)
- **Compiler**: g++ with C++11, -O3 optimization

---

## Performance Results

### Execution Time Comparison

| Metric | Logistic Regression | Decision Tree | Winner |
|--------|-------------------|---------------|---------|
| **Load Time** | 0.0590 seconds | 0.0066 seconds | ✅ Decision Tree (9x faster) |
| **Training Time** | 0.2884 seconds | 0.0726 seconds | ✅ Decision Tree (4x faster) |
| **Evaluation Time** | 0.0013 seconds | 0.0002 seconds | ✅ Decision Tree (6.5x faster) |
| **Total Time** | 0.3496 seconds | 0.0799 seconds | ✅ Decision Tree (4.4x faster) |

**Key Insight**: Decision Tree is significantly faster in all phases, with an overall speedup of **4.4x** compared to Logistic Regression.

### Accuracy and Prediction Comparison

| Metric | Logistic Regression | Decision Tree | Analysis |
|--------|-------------------|---------------|----------|
| **Accuracy** | 100.00% | 49.55% | Logistic Regression significantly better |
| **Predicted Death Rate** | 0.005% | 19.15% | Dramatically different predictions |

---

## Detailed Analysis

### 1. Training Speed

**Decision Tree Wins**: The Decision Tree algorithm demonstrates superior training speed with a 4x advantage over Logistic Regression.

- **Logistic Regression**: 0.2884 seconds
  - Requires iterative gradient descent optimization
  - Multiple passes through the data to converge
  - Computational complexity: O(n × m × iterations)

- **Decision Tree**: 0.0726 seconds
  - Non-iterative recursive splitting approach
  - Single pass through data at each level
  - Computational complexity: O(n × m × log n)

### 2. Data Loading

**Decision Tree Wins**: The Decision Tree shows 9x faster data loading.

- **Logistic Regression**: 0.0590 seconds
  - Likely includes feature encoding/normalization
  - Additional preprocessing for gradient descent

- **Decision Tree**: 0.0066 seconds
  - Direct use of categorical features
  - Minimal preprocessing required

### 3. Prediction Accuracy

**Logistic Regression Wins**: Achieves perfect 100% accuracy, while Decision Tree achieves only 49.55%.

**Analysis of Results**:

1. **Logistic Regression (100% accuracy)**:
   - The extremely high accuracy with near-zero death rate (0.005%) suggests potential issues:
     - Possible overfitting to training data
     - May be predicting "survival" for almost all cases
     - With gradient descent, might have converged to a solution that favors the majority class
   - The low death rate (0.005%) seems unrealistically optimistic for clinical data

2. **Decision Tree (49.55% accuracy)**:
   - Approximately random performance (near 50% for binary classification)
   - Predicts 19.15% death rate, which may be more realistic for clinical data
   - Possible explanations:
     - Limited tree depth causing underfitting
     - Features may not have strong predictive power for tree-based splitting
     - May be overfitting to training set noise at leaf nodes

### 4. Model Characteristics

| Characteristic | Logistic Regression | Decision Tree |
|----------------|-------------------|---------------|
| **Model Type** | Linear classifier | Non-linear classifier |
| **Interpretability** | Coefficients per feature | Clear decision rules |
| **Feature Processing** | Requires encoding/scaling | Works with raw categorical |
| **Overfitting Risk** | Lower (with regularization) | Higher (without pruning) |
| **Training Algorithm** | Iterative (Gradient Descent) | Greedy recursive splitting |
| **Memory Usage** | Minimal (only weights) | Higher (stores tree structure) |

---

## When to Use Each Algorithm

### Use Logistic Regression When:
✅ You need probabilistic outputs for risk assessment  
✅ The relationship between features and outcome is approximately linear  
✅ You want a simpler, more stable model  
✅ You need feature importance through coefficients  
✅ Training time is not critical  
✅ You have proper regularization to prevent overfitting  

### Use Decision Tree When:
✅ You need interpretable decision rules for clinical decisions  
✅ The relationship is non-linear or has complex interactions  
✅ You want very fast training times  
✅ Feature interactions are important  
✅ You can implement proper pruning to prevent overfitting  
✅ Working with raw categorical features without preprocessing  

---

## Recommendations for Improvement

### For Logistic Regression:
1. **Add Regularization**: Implement L1 or L2 regularization to prevent overfitting
2. **Cross-Validation**: Use k-fold cross-validation to better assess true performance
3. **Class Balancing**: Address class imbalance if present in the dataset
4. **Learning Rate Tuning**: Optimize learning rate for better convergence

### For Decision Tree:
1. **Pruning**: Implement post-pruning to reduce overfitting
2. **Hyperparameter Tuning**: 
   - Increase max depth if tree is too shallow
   - Adjust min samples split threshold
   - Tune minimum samples per leaf
3. **Feature Engineering**: Create more informative features or feature interactions
4. **Ensemble Methods**: Consider Random Forest or Gradient Boosting for better accuracy
5. **Cross-Validation**: Use proper train/test split or k-fold validation

### General Improvements:
1. **Stratified Splitting**: Ensure train/test sets have balanced class distributions
2. **Performance Metrics**: Add precision, recall, F1-score, and AUC-ROC
3. **Confusion Matrix**: Visualize true positives, false positives, etc.
4. **Feature Analysis**: Analyze feature importance for both algorithms
5. **Dataset Analysis**: Investigate the actual death rate in the dataset

---

## Parallel Implementation Potential

Both algorithms can benefit from parallelization:

### Logistic Regression Parallelization:
- Gradient computation across data samples (OpenMP, MPI)
- Mini-batch processing (distribute batches across threads/processes)
- CUDA for matrix operations on large datasets

### Decision Tree Parallelization:
- Parallel split evaluation across features
- Parallel prediction for multiple patients
- Ensemble methods (Random Forest) with parallel tree building

**Expected Impact**: For the Decision Tree's already fast training time (0.07s), parallelization may have limited benefit on small datasets but would be valuable for larger datasets (>100K records).

---

## Conclusion

**Performance Winner**: ✅ **Decision Tree** - 4.4x faster overall execution  
**Accuracy Winner**: ✅ **Logistic Regression** - 100% accuracy vs 49.55%

**Overall Assessment**: 
- **Logistic Regression** is currently the better choice for this dataset due to dramatically superior accuracy, though the perfect accuracy suggests potential overfitting that should be investigated.
- **Decision Tree** offers impressive speed advantages but requires significant improvements in accuracy through hyperparameter tuning and pruning.
- Both algorithms would benefit from proper cross-validation, regularization/pruning, and more robust evaluation metrics.

**Next Steps**:
1. Investigate the suspiciously perfect accuracy of Logistic Regression
2. Implement proper train/test split and cross-validation for both algorithms
3. Add pruning to Decision Tree implementation
4. Compare on multiple performance metrics (precision, recall, F1, AUC)
5. Consider ensemble methods to combine strengths of both approaches

---

## How to Reproduce These Results

```bash
# Navigate to the project directory
cd deathPrediction

# Build both implementations
make serial        # Logistic Regression
make dt_serial     # Decision Tree

# Run Logistic Regression
./serial_death_pred mimic_data.csv

# Run Decision Tree
./serial_decision_tree_death_pred mimic_data.csv
```

## References

- Decision Tree Implementation: [DECISION_TREE_README.md](DECISION_TREE_README.md)
- Project Overview: [../README.md](../README.md)
- Hybrid Implementations: [HYBRID_RESULTS.md](HYBRID_RESULTS.md)
- CUDA Performance: [CUDA_PERFORMANCE_COMPARISON.md](CUDA_PERFORMANCE_COMPARISON.md)
