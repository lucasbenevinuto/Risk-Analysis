# Financial Loan Risk Assessment and Approval Prediction

This project aims to develop predictive models for assessing financial loan risk and determining loan approval likelihood using a synthetic dataset. We built both regression and classification models, leveraging Bagging with Decision Tree algorithms to predict a continuous risk score and a binary loan approval status.

## Dataset Overview

The dataset consists of 20,000 records containing personal and financial data, which provide a strong foundation for risk assessment and loan approval modeling. The dataset includes 35 features such as demographic information, credit history, income levels, and other financial metrics.

### Key Features

- **ApplicationDate**: Loan application date
- **Age**: Applicant's age
- **AnnualIncome**: Yearly income
- **CreditScore**: Creditworthiness score
- **EmploymentStatus**: Job situation
- **EducationLevel**: Highest education attained
- **LoanAmount**: Requested loan size
- **LoanDuration**: Loan repayment period
- **DebtToIncomeRatio**: Debt to income proportion
- **BankruptcyHistory**: Bankruptcy records
- **LoanPurpose**: Reason for loan
- **PreviousLoanDefaults**: Prior loan defaults
- **RiskScore**: Risk assessment score (target for regression)
- **LoanApproved**: Loan approval status (target for classification)

And many more, providing a comprehensive overview of each applicant’s financial background.

## Project Objectives

1. **Risk Score Regression**: Predict the likelihood of loan default or financial instability by estimating a continuous risk score.
   
2. **Loan Approval Classification**: Develop a binary classification model to determine whether a loan applicant is likely to be approved or denied.

## Methods and Models

### Data Preprocessing

- Handled missing values
- Scaled continuous variables
- Encoded categorical features (e.g., EmploymentStatus, EducationLevel)
- Split data into training and testing sets

### Predictive Models

1. **Classification (Loan Approval Prediction)**
   - Algorithm: **Bagging with DecisionTreeClassifier**
   - Target: `RiskClassification`
   - Evaluation Metrics: Accuracy, Precision, Recall, F1-score

2. **Regression (Risk Score Prediction)**
   - Algorithm: **Bagging with DecisionTreeRegressor**
   - Target: `RiskScore`
   - Evaluation Metrics: Mean Squared Error (MSE), R-squared (R²)

### Model Evaluation

- Both models were evaluated using appropriate metrics (classification and regression).
- Cross-validation was applied to ensure robustness.

## Results

- **Classification Model**: Achieved an accuracy of %, with an ROC-AUC score of 0.91, indicating good performance in predicting loan approval.
- **Regression Model**: Achieved an R² of 0.79 and MSE of 0.15 in predicting the risk score, demonstrating the model's ability to assess financial risk effectively.

## Repository Structure

```bash
├── data/
│   └── financial_loan_risk_dataset.csv  # Dataset used
├── notebooks/
│   └── Analysis.ipynb       # EDA and visualization notebook
│   └── BagginClassfier.ipynb
│   └── BagginRegression.ipynb             # Model training and evaluation notebook
├── models/
│   └── BagginClassfie.pkl          # Saved classification model
│   └── BagginRegression.pkl             # Saved regression model
├── README.md                            # Project documentation
└── requirements.txt                     # List of dependencies
```

## Installation and Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/username/loan-risk-assessment.git
   cd loan-risk-assessment
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebooks for Exploratory Data Analysis and Model Building:
   ```bash
   jupyter notebook
   ```

## Future Work

- Explore other ensemble techniques such as Random Forest and Gradient Boosting.
- Test the models with real-world datasets for performance validation.
- Implement a web interface for real-time loan risk prediction.
