# Task 5 - Decision Trees and Random Forests

## Objective
Build and compare Decision Tree and Random Forest models for heart disease classification.

## Dataset
Heart disease dataset saved as `datasets/heart_disease_dataset.csv`.

## Key Steps

1. Loaded dataset and inspected features
2. Split into train/test sets
3. Trained a Decision Tree classifier
4. Visualized the Decision Tree structure
5. Evaluated overfitting using deep and shallow trees
6. Trained a Random Forest classifier
7. Compared Decision Tree and Random Forest accuracy
8. Interpreted feature importance
9. Generated a confusion matrix and classification report
10. Verified stability with 5-fold cross-validation

## Results
- Random Forest achieved higher test accuracy than the single Decision Tree.
- A deep Decision Tree exhibited overfitting when train accuracy was near 100% and test accuracy dropped.
- A shallow Decision Tree reduced overfitting while preserving generalization.
- Important predictors included the features with the highest Random Forest importance scores.
- Cross-validation confirmed stable model performance across folds.

## Conclusion
Random Forest is more robust than a single Decision Tree for this classification task, especially in terms of generalization and overfitting control. The model outputs include a tree visualization, feature importance plot, confusion matrix, and cross-validation summary.
