# Imputation Strategies for Age

This document outlines various methods for handling missing values in the 'Age' column, which is crucial for accurate model training.

## 1. Simple Imputation Methods

These methods are quick to implement but can underestimate the variance and introduce bias.

### A. Mean Imputation

- **Method:** Replace all missing values with the average age of the existing data.
- **Pros:** Simple, fast.
- **Cons:** Reduces the variance of the feature, potentially skewing the distribution towards the mean.

### B. Median Imputation

- **Method:** Replace all missing values with the median age.
- **Pros:** More robust to outliers than the mean.
- **Cons:** Still reduces variance.

### C. Constant Value Imputation

- **Method:** Replace missing values with a constant (e.g., 0 or a specific placeholder value).
- **Pros:** Simple.
- **Cons:** Can be misleading if the constant value has no inherent meaning.

## 2. Advanced Imputation Methods (Recommended)

These methods leverage relationships between features to create more accurate estimates.

### A. Regression Imputation

- **Method:** Use other features (e.g., 'Pclass', 'SibSp', 'Parch') as predictors in a regression model to predict the missing age.
- **Pros:** Captures the relationship between age and other variables, leading to more accurate estimates.
- **Cons:** Requires careful feature selection and model training.

### B. Model-Based Imputation (e.g., MICE)

- **Method:** Multivariate Imputation by Chained Equations (MICE) iteratively models each feature with missing values using the other features.
- **Pros:** Considered one of the most statistically sound methods, as it accounts for the correlation structure across all variables.
- **Cons:** Computationally intensive.

## 3. Title/Prefix-Based Imputation (Advanced)

A highly effective strategy is to use the passenger's title (derived from the name) to infer a more accurate age estimate, as titles often correlate strongly with age groups.

**Logic:**

1.  Extract the title (e.g., Mr., Miss., Master., Rev., etc.) from the passenger's name.
2.  Map this title to a representative age group or median age.
3.  Use this inferred age as the imputation value.

**Example Age Mapping Table (Hypothetical):**

| Title Prefix | Implied Age Group | Representative Age (Median) |
| :----------- | :---------------- | :-------------------------- |
| **Master**   | Young Boy         | 2 - 10 years                |
| **Miss**     | Young Woman       | 15 - 30 years               |
| **Mrs**      | Married Woman     | 25 - 45 years               |
| **Mr**       | Adult Male        | 30 - 55 years               |

**Note:** This method requires cleaning the 'Name' column to reliably extract the title first.

## Summary Table

| Method     | Bias Reduction | Complexity | Recommendation                      |
| :--------- | :------------- | :--------- | :---------------------------------- |
| Mean       | High           | Low        | Baseline check.                     |
| Median     | Medium         | Low        | Good alternative to Mean.           |
| Regression | Low            | Medium     | Best for capturing relationships.   |
| MICE       | Lowest         | High       | Best practice for complex datasets. |
