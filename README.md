# Credit Risk Modeling Application

This project provides end-to-end credit risk analysis including data cleaning, feature engineering, model training, and performance evaluation.



### Key Features

- **Data Analysis**: Exploratory data analysis with interactive visualisations
- **Data Cleaning**: Automated pipeline for handling missing values, outliers, and categorical encoding
- **Feature Engineering**: Creation of risk-relevant features 
- **Model Training**: Implementation of XGBoost and Logistic Regression with hyperparameter tuning
- **Model Evaluation**: Performance comparison, feature importance analysis, and SHAP explanations


## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download the Repository

```bash
# If using git
git clone <repository-url>
cd credit-risk-modeling

# Or download the files directly and navigate to the directory
```

### Step 2: Install Required Packages

```bash
pip install -r requirements.txt
```

**Alternative: Create a Virtual Environment (Recommended)**

```bash
# Create virtual environment
python -m venv credit_risk_env

# Activate virtual environment
# On Windows:
credit_risk_env\Scripts\activate
# On macOS/Linux:
source credit_risk_env/bin/activate

# Install requirements
pip install -r requirements.txt
```


## Running the Application

### Start the Streamlit App

```bash
streamlit run credit_risk_app.py
```

**Note**: Replace `credit_risk_app.py` with the actual name of your Python script.

### Access the Application

After running the command, Streamlit will automatically open your default web browser and navigate to:

```
http://localhost:8501
```

If it doesn't open automatically, you can manually navigate to this URL.

## Application Usage

### 1. Credit Risk Analysis Page

- **Dataset Overview**: View comprehensive statistics about your loan dataset
- **Visualizations**: Interactive charts showing default rates by various factors
- **Correlation Analysis**: Heatmap showing relationships between features
- **Key Insights**: Summary of important risk factors

### 2. Model Pipeline Page

The model pipeline follows these steps:

1. **Data Cleaning**
   - Remove duplicates
   - Handle missing values
   - Remove extreme outliers
   - Encode categorical variables

2. **Data Splitting**
   - Split data into training (60%), validation (20%), and test (20%) sets
   - Ensure stratified sampling to maintain class balance

3. **Feature Engineering**
   - Create debt-to-income ratios
   - Generate age and income groups
   - Build interaction features

4. **Model Training**
   - **XGBoost**: Gradient boosting with hyperparameter optimisation
   - **Logistic Regression**: Linear model with regularisation
   - Cross-validation for robust performance estimation

5. **Model Evaluation**
   - ROC-AUC scores for performance comparison
   - Feature importance analysis
   - SHAP values for model interpretability
   - Overfitting detection

## Key Technical Features


### Model Interpretability

- **Feature Importance**: Shows which factors most influence predictions
- **SHAP Values**: Explains individual predictions and feature contributions
- **Coefficient Analysis**: For logistic regression, shows positive/negative impacts
