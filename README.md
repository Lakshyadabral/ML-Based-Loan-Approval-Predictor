# ML-Based Loan Approval Predictor

## Project Overview
**ML-Based Loan Approval Predictor** is a machine learning–based application designed to predict whether a loan application should be **Approved** or **Rejected** based on an applicant’s personal, financial, and credit information.

The project simulates a real-world scenario where a mid-sized financial institution (SecureTrust Bank) aims to reduce:
-  Rejection of low-risk customers, leading to loss of business 
-  Approval of high-risk customers, leading to financial losses 

by introducing a **data-driven loan screening system** to assist human loan officers.

---

## Problem Statement
Traditionally, loan approvals were handled via **manual verification**, which is:
- Time-consuming  
- Inconsistent  
- Prone to human bias  

The objective of this project is to:
> Build a machine learning model that learns patterns from historical loan data and predicts loan approval decisions efficiently and consistently before final human verification.

---

## Dataset Description
Each row in the dataset represents a **loan applicant**, with features describing their financial, demographic, and credit profile.

### Key Features
| Feature | Description |
|------|------------|
| Applicant_Income | Monthly income of applicant |
| Coapplicant_Income | Monthly income of co-applicant |
| Employment_Status | Salaried / Self-Employed / Business |
| Age | Applicant age |
| Marital_Status | Married / Single |
| Dependents | Number of dependents |
| Credit_Score | Credit bureau score (300–900) |
| Existing_Loans | Number of running loans |
| DTI_Ratio | Debt-to-Income ratio |
| Savings | Savings balance |
| Collateral_Value | Collateral value |
| Loan_Amount | Requested loan amount |
| Loan_Term | Loan duration (months) |
| Loan_Purpose | Home / Education / Personal / Business |
| Property_Area | Urban / Semi-Urban / Rural |
| Education_Level | Undergraduate / Graduate / Postgraduate |
| Gender | Male / Female |
| Employer_Category | Govt / Private / Self |
| Loan_Approved | **Target variable** (Approved / Rejected) |

---

## Machine Learning Approach

### Exploratory Data Analysis & Model Selection

The complete exploratory data analysis (EDA), feature inspection, and comparison of multiple machine learning models
(Logistic Regression, KNN, and Naive Bayes) is documented in the following Jupyter notebook:

**`analysis_eda_model_comparison.ipynb`**

This notebook details:
- Data distributions and class balance
- Feature relationships and correlations
- Evaluation of multiple models using Precision, Recall, F1-score, and Accuracy
- The rationale for selecting Logistic Regression as the final deployed model based on balanced performance

### Models Studied
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Naive Bayes

### Final Model Choice
**Logistic Regression** was selected for deployment because:
- It provided the **best balance of Precision, Recall, and F1-score**
- It handled **one-hot encoded categorical data** well
- It offers **interpretability**, which is critical in financial decision-making
- It aligns with real-world credit scoring practices
Precision alone was not sufficient for this problem, as the objective was to minimize false rejections of eligible applicants while still controlling risk, making F1-score a more appropriate evaluation metric.


---

## Data Preprocessing
- Missing numerical values → Mean Imputation  
- Missing categorical values → Most Frequent Imputation  
- Categorical variables → One-Hot Encoding  
- Numerical features → Standard Scaling  
- Identifier columns removed (`Applicant_ID`)  
- Target rows with missing labels removed  

All preprocessing and modeling steps are combined into a **single Scikit-learn Pipeline** to ensure:
- No data leakage  
- Reproducible training  
- Reliable deployment  

---

## Application Interface (Streamlit)
A **Streamlit web app** allows users to:
- Enter applicant and loan details via a form
- Receive instant predictions (Approved / Rejected)
- View model confidence scores

The UI is **fully custom-built** using Streamlit components and mirrors the features used during training.

---

## Limitations
- The dataset is educational and does not reflect jurisdiction-specific lending policies (e.g., Canadian GDS/TDS ratios).
- The model learns from historical approval patterns and may replicate dataset bias.
- This system is intended as a decision-support tool and should not be used for real-world financial decisions.


## How to Run the Project

### Install dependencies
```bash
pip install -r requirements.txt
