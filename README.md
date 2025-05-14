# Home-credit-default-risk-

This repository contains the final project for the *Data Mining* course at Zhejiang University, focused on predicting the creditworthiness of clients using real-world financial data provided by Home Credit.
This project attempts to improve loan default prediction models using machine learning techniques on a dataset provided through the [Home Credit Default Risk Kaggle competition](https://www.kaggle.com/competitions/home-credit-default-risk/data).

## Dataset

The dataset was sourced from Kaggle competition. The following files were used:

- `application_test.csv`: Main application dataset.
- `bureau.csv`: Credit history from other institutions.
- `credit_card_balance.csv`: Behavior and credit usage trends.
- `installments_payments.csv`: Repayment history.
- `previous_application.csv`: History of previous loan applications.

## Data Preprocessing

Several preprocessing steps were carried out:

- **Missing Value Handling**: Mean/median imputation for continuous variables.
- **Normalization & Encoding**: Standardized continuous variables, one-hot encoding for categorical variables.
- **Outlier Detection**: Used interquartile range and domain thresholds.
- **Data Integration**: Merged datasets via unique client/application identifiers.
- **Multicollinearity Check**: Correlation matrix analysis revealed no serious issues.

## Feature Engineering

New features were derived to enhance model performance, based on common practices:

- `Credit/Annuity Ratio`: Indicates borrower preference for smaller or longer-term payments.
- `Late Payment`: Indicator for history of missed repayments.
- `Credit Utilization`: Suggests financial discipline or strain.
- `Debt Ratio`: Shows proportion of income used for debt repayment.
- `Interest Rate`: Reflects perceived borrower risk.
- `Goods/Price Ratio`: Assesses if loan usage aligns with purchase value.

## Modeling

Implementation and comparison of several machine learning models:

| Model              | Strengths                                            | Weaknesses                            |
|-------------------|------------------------------------------------------|---------------------------------------|
| XGBoost           | High accuracy, handles missing values automatically  | Slower on very large datasets         |
| LightGBM          | Fast, supports categorical features                  | May overfit on small data             |
| GridSearch + LGBM | Optimized hyperparameters                            | Computationally expensive             |
| Random Forest     | Reduces overfitting                                  | Less interpretable and slower         |

 

### Best Models: 

### XGBoost
- **ROC AUC (Using all Features)**: 77.29%
- **ROC AUC (MUsing all features and complete Dataset)**: 78.09%
  
- **Top Features**:
  - `EXT_SOURCE_3`
  - `EXT_SOURCE_2`
  - `DAYS_BIRTH`
  - `DEBT_RATIO`
  - `AMT_CREDIT`


