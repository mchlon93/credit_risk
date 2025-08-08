import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
import xgboost as xgb
import shap
import time

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Credit Risk Modeling",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS  
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #016FD0;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2.5rem;
        color: #013968;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .subsection-header {
        font-size: 1.5rem;
        color:#36454F;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 2px solid #016FD0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }

    
    .stButton > button[kind="primary"] {
        background-color: #016FD0 !important;
        color: white !important;
        border: none !important;
        border-radius: 0.5rem !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #014a94 !important;
        box-shadow: 0 4px 8px rgba(1, 111, 208, 0.3) !important;
        transform: translateY(-1px) !important;
    }
    
    .stButton > button[kind="primary"]:active {
        background-color: #013968 !important;
        transform: translateY(0px) !important;
    }
    
    .stButton > button[kind="primary"]:focus:not(:active) {
        border: 2px solid #016FD0 !important;
        box-shadow: 0 0 0 0.2rem rgba(1, 111, 208, 0.25) !important;
    }
</style>
""", unsafe_allow_html=True)

############# DATA CLEANING PIPELINE #############

class DataCleaningPipeline:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputers = {}
        
    def clean_data(self, df, target_col='loan_status'):
        """Complete data cleaning pipeline"""
        cleaning_log = []
        df_clean = df.copy()
        
        # 1. Remove duplicates
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        duplicates_removed = initial_rows - len(df_clean)
        if duplicates_removed > 0:
            cleaning_log.append(f"✓ Removed {duplicates_removed} duplicate rows")
        else:
            cleaning_log.append("✓ No duplicate rows found")
        
        # 2. Handle missing values
        missing_cols = df_clean.columns[df_clean.isnull().any()].tolist()
        if missing_cols:
            for col in missing_cols:
                missing_count = df_clean[col].isnull().sum()
                missing_pct = (missing_count / len(df_clean)) * 100
                
                if missing_pct > 50:
                    df_clean = df_clean.drop(columns=[col])
                    cleaning_log.append(f"× Dropped column '{col}' (>{missing_pct:.1f}% missing)")
                else:
                    if df_clean[col].dtype in ['int64', 'float64']:
                        # Numerical columns - use median
                        imputer = SimpleImputer(strategy='median')
                        df_clean[col] = imputer.fit_transform(df_clean[[col]]).ravel()
                        self.imputers[col] = imputer
                        cleaning_log.append(f"→ Imputed {missing_count} missing values in '{col}' with median")
                    else:
                        # Categorical columns - use mode
                        imputer = SimpleImputer(strategy='most_frequent')
                        df_clean[col] = imputer.fit_transform(df_clean[[col]]).ravel()
                        self.imputers[col] = imputer
                        cleaning_log.append(f"→ Imputed {missing_count} missing values in '{col}' with mode")
        else:
            cleaning_log.append("✓ No missing values found")
        
        # 3. Handle outliers (using IQR method for numerical columns)
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numerical_cols:
            numerical_cols.remove(target_col)
        
        outliers_removed = 0
        for col in numerical_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR  # Using 3*IQR for extreme outliers only
            upper_bound = Q3 + 3 * IQR
            
            initial_count = len(df_clean)
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
            removed = initial_count - len(df_clean)
            outliers_removed += removed
            
            if removed > 0:
                cleaning_log.append(f"→ Removed {removed} extreme outliers from '{col}'")
        
        if outliers_removed == 0:
            cleaning_log.append("✓ No extreme outliers found")
        
        # 4. Encode categorical variables
        categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            self.label_encoders[col] = le
            unique_vals = len(le.classes_)
            cleaning_log.append(f"→ Encoded categorical column '{col}' ({unique_vals} unique values)")
        
        # 5. Final data info
        cleaning_log.append(f"✓ Final dataset shape: {df_clean.shape}")
        cleaning_log.append(f"✓ Data cleaning completed successfully!")
        
        return df_clean, cleaning_log

############# FEATURE ENGINEERING PIPELINE ############# 

class FeatureEngineeringPipeline:
    def __init__(self):
        self.fitted_params = {}
        self.is_fitted = False
    
    def fit(self, X_train):        
        # Calculate statistics from training data only
        self.fitted_params = {
            'income_quartiles': X_train['person_income'].quantile([0.25, 0.5, 0.75]),
            'high_rate_threshold': X_train['loan_int_rate'].quantile(0.75),
            'debt_to_income_median': (X_train['loan_amnt'] / X_train['person_income']).median(),
            'credit_hist_median': X_train['cb_person_cred_hist_length'].median()
        }
        
        self.is_fitted = True
        return self
    
    def transform(self, df):
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        df_features = df.copy()
        feature_descriptions = {}
        
        # 1. Debt-to-Income Ratio
        df_features['debt_to_income'] = df_features['loan_amnt'] / df_features['person_income']
        feature_descriptions['debt_to_income'] = "Loan amount divided by annual income - higher values indicate higher risk"
        
        # 2. Age groups  
        df_features['age_group'] = pd.cut(df_features['person_age'], 
                                        bins=[0, 25, 35, 50, 100], 
                                        labels=[0, 1, 2, 3])
        df_features['age_group'] = df_features['age_group'].astype(int)
        feature_descriptions['age_group'] = "Age categorized into groups: Young (≤25), Adult (26-35), Middle Age (36-50), Senior (>50)"
        
        # 3. Income groups  
        income_quartiles = self.fitted_params['income_quartiles']
        df_features['income_group'] = pd.cut(df_features['person_income'],
                                           bins=[0, income_quartiles[0.25], income_quartiles[0.5], 
                                                income_quartiles[0.75], float('inf')],
                                           labels=[0, 1, 2, 3])
        df_features['income_group'] = df_features['income_group'].astype(int)
        feature_descriptions['income_group'] = "Income categorized into quartiles based on training data: Low, Medium, High, Very High"
        
        # 4. Employment stability
        df_features['employment_stable'] = (df_features['person_emp_length'] >= 2).astype(int)
        feature_descriptions['employment_stable'] = "Binary feature: 1 if employment length ≥ 2 years, 0 otherwise"
        
        # 5. High interest rate flag  
        high_rate_threshold = self.fitted_params['high_rate_threshold']
        df_features['high_interest_rate'] = (df_features['loan_int_rate'] > high_rate_threshold).astype(int)
        feature_descriptions['high_interest_rate'] = f"Binary feature: 1 if interest rate > {high_rate_threshold:.2f}% (75th percentile from training), 0 otherwise"
        
        # 6. Credit history score
        df_features['credit_score'] = df_features['cb_person_cred_hist_length'] / df_features['person_age']
        feature_descriptions['credit_score'] = "Credit history length divided by age - indicates how early person started building credit"
        
        # 7. Loan amount relative to income percentile  
        df_features['loan_amount_percentile'] = df_features.groupby('income_group')['loan_amnt'].rank(pct=True)
        feature_descriptions['loan_amount_percentile'] = "Loan amount percentile within same income group"
        
        # 8. Risk interaction features  
        debt_to_income_median = self.fitted_params['debt_to_income_median']
        df_features['high_debt_young'] = ((df_features['debt_to_income'] > debt_to_income_median) & 
                                        (df_features['person_age'] < 30)).astype(int)
        feature_descriptions['high_debt_young'] = f"Binary feature: 1 if debt-to-income > {debt_to_income_median:.3f} (training median) AND age < 30"
        
        # 9. Home ownership credit interaction  
        credit_hist_median = self.fitted_params['credit_hist_median']
        df_features['own_home_good_credit'] = (df_features['person_home_ownership'] * 
                                             (df_features['cb_person_cred_hist_length'] > credit_hist_median)).astype(int)
        feature_descriptions['own_home_good_credit'] = f"Interaction between home ownership and good credit history (>{credit_hist_median:.1f} years)"
        
        return df_features, feature_descriptions
    
    def fit_transform(self, X_train):
        """Fit on training data and transform it"""
        return self.fit(X_train).transform(X_train)

############# MODEL TRAINING PIPELINE #############

class ModelTrainingPipeline:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = None
        
    def split_data(self, X, y, test_size=0.2, val_size=0.2, random_state=42):
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_xgboost(self, X_train, X_val, X_test, y_train, y_val, y_test):
        
        # Define hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        # Initialize model
        xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1
        )
        
        # Fit the model
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_xgb = grid_search.best_estimator_
        
        # Predictions
        y_pred_train = best_xgb.predict(X_train)
        y_pred_val = best_xgb.predict(X_val)
        y_pred_test = best_xgb.predict(X_test)
        
        y_pred_proba_train = best_xgb.predict_proba(X_train)[:, 1]
        y_pred_proba_val = best_xgb.predict_proba(X_val)[:, 1]
        y_pred_proba_test = best_xgb.predict_proba(X_test)[:, 1]
        
        # Metrics
        results = {
            'model': best_xgb,
            'best_params': grid_search.best_params_,
            'train_auc': roc_auc_score(y_train, y_pred_proba_train),
            'val_auc': roc_auc_score(y_val, y_pred_proba_val),
            'test_auc': roc_auc_score(y_test, y_pred_proba_test),
            'classification_report': classification_report(y_test, y_pred_test),
            'confusion_matrix': confusion_matrix(y_test, y_pred_test),
            'predictions': y_pred_test,
            'probabilities': y_pred_proba_test
        }
        
        self.models['xgboost'] = results
        return results
    
    def train_logistic_regression(self, X_train, X_val, X_test, y_train, y_val, y_test):        
        # Scale features for logistic regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['logistic'] = scaler
        
        # Define hyperparameter grid
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000]
        }
        
        # Initialize model
        lr_model = LogisticRegression(random_state=42)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            lr_model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1
        )
        
        # Fit the model
        grid_search.fit(X_train_scaled, y_train)
        
        # Best model
        best_lr = grid_search.best_estimator_
        
        # Predictions
        y_pred_train = best_lr.predict(X_train_scaled)
        y_pred_val = best_lr.predict(X_val_scaled)
        y_pred_test = best_lr.predict(X_test_scaled)
        
        y_pred_proba_train = best_lr.predict_proba(X_train_scaled)[:, 1]
        y_pred_proba_val = best_lr.predict_proba(X_val_scaled)[:, 1]
        y_pred_proba_test = best_lr.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        results = {
            'model': best_lr,
            'best_params': grid_search.best_params_,
            'train_auc': roc_auc_score(y_train, y_pred_proba_train),
            'val_auc': roc_auc_score(y_val, y_pred_proba_val),
            'test_auc': roc_auc_score(y_test, y_pred_proba_test),
            'classification_report': classification_report(y_test, y_pred_test),
            'confusion_matrix': confusion_matrix(y_test, y_pred_test),
            'predictions': y_pred_test,
            'probabilities': y_pred_proba_test,
            'coefficients': best_lr.coef_[0],
            'intercept': best_lr.intercept_[0]
        }
        
        self.models['logistic'] = results
        return results

############# VISUALIZATION FUNCTIONS #############

def create_eda_visualizations(df):    
    # Default rate by categorical features
    categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=categorical_cols,
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    for i, col in enumerate(categorical_cols):
        if col in df.columns:
            default_rates = df.groupby(col)['loan_status'].mean().reset_index()
            default_rates['loan_status'] = default_rates['loan_status'] * 100
            
            row = i // 2 + 1
            col_pos = i % 2 + 1
            
            fig.add_trace(
                go.Bar(x=default_rates[col], y=default_rates['loan_status'],
                      name=col, showlegend=False),
                row=row, col=col_pos
            )
    
    fig.update_layout(height=600, title_text="Default Rates by Categorical Features")
    fig.update_yaxes(title_text="Default Rate (%)")
    
    return fig

def create_correlation_heatmap(df):
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numerical_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0
    ))
    
    fig.update_layout(
        title="Feature Correlation Matrix",
        width=800,
        height=600
    )
    
    return fig

def plot_feature_importance(model, feature_names, title):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = np.abs(model.coef_[0])
    
    # Get top 15 features
    indices = np.argsort(importances)[::-1][:15]
    
    fig = go.Figure(data=go.Bar(
        x=[feature_names[i] for i in indices],
        y=importances[indices]
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Features",
        yaxis_title="Importance",
        xaxis_tickangle=-45
    )
    
    return fig

def plot_roc_curves(models, X_test, y_test):
    fig = go.Figure()
    
    for model_name, model_results in models.items():
        if model_name == 'logistic':
            # Use scaled data for logistic regression
            y_proba = model_results['probabilities']
        else:
            y_proba = model_results['probabilities']
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = model_results['test_auc']
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{model_name.title()} (AUC = {auc_score:.3f})'
        ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(dash='dash')
    ))
    
    fig.update_layout(
        title='ROC Curves Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=600,
        height=500
    )
    
    return fig

def plot_shap_values(model, X_test, feature_names):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    
    # Create a summary plot with smaller figure size
    fig, ax = plt.subplots(figsize=(8, 6))  # Reduced from (10, 8)
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False, max_display=15)
    plt.tight_layout()
    
    return fig

def plot_logistic_coefficients(model, feature_names):
    coefficients = model.coef_[0]
    
    # Sort by absolute value
    indices = np.argsort(np.abs(coefficients))[::-1][:15]
    
    fig = go.Figure(data=go.Bar(
        x=[feature_names[i] for i in indices],
        y=coefficients[indices],
        marker_color=['red' if x < 0 else 'blue' for x in coefficients[indices]]
    ))
    
    fig.update_layout(
        title="Logistic Regression Coefficients (Top 15 Features)",
        xaxis_title="Features",
        yaxis_title="Coefficient Value",
        xaxis_tickangle=-45
    )
    
    return fig


############# STREAMLIT APP #############

def main():
    st.markdown('<h1 class="main-header"> Credit Risk Modeling</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Credit Risk Analysis", "Model Pipeline"])
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'data_cleaned' not in st.session_state:
        st.session_state.data_cleaned = False
    if 'features_created' not in st.session_state:
        st.session_state.features_created = False
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = {'xgboost': False, 'logistic': False}
    
    # Data upload section - load from file directly
    try:
        df = pd.read_csv('credit_risk_dataset.csv')
        st.session_state.original_data = df
        st.session_state.data_loaded = True
    except FileNotFoundError:
        st.sidebar.error("credit_risk_dataset.csv not found. Please ensure the file is in the same directory as this script.")
        st.session_state.data_loaded = False
    except Exception as e:
        st.sidebar.error(f"Error loading data: {str(e)}")
        st.session_state.data_loaded = False
    
    if not st.session_state.data_loaded:
        st.warning("Please ensure 'credit_risk_dataset.csv' is in the same directory as this script.")
        st.info("Expected columns: person_age, person_income, person_home_ownership, person_emp_length, loan_intent, loan_grade, loan_amnt, loan_int_rate, loan_status, loan_percent_income, cb_person_default_on_file, cb_person_cred_hist_length")
        return
    
    df = st.session_state.original_data
    
    ############# CREDIT RISK ANALYSIS PAGE #############
    
    if page == "Credit Risk Analysis":
        st.markdown('<h2 class="section-header">Credit Risk Analysis</h2>', unsafe_allow_html=True)
        
        # Dataset explanation
        st.markdown("""
        ### Dataset Overview
        
        This application analyzes a **loan default prediction dataset** containing information about borrowers and their loan characteristics. 
        The dataset includes demographic information, employment details, loan specifications, and historical credit data to predict 
        the likelihood of loan default.
        
        **Key Features:**
        - **Personal Information**: Age, income, employment length, home ownership
        - **Loan Details**: Amount, interest rate, purpose, grade, percent of income
        - **Credit History**: Previous defaults, credit history length
        - **Target Variable**: Loan status (0 = No Default, 1 = Default)
        """)
        
        # Dataset statistics in boxes
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h4>Total Records</h4>
                <h2>{:,}</h2>
            </div>
            """.format(len(df)), unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4>Default Rate</h4>
                <h2>{:.2%}</h2>
            </div>
            """.format(df['loan_status'].mean()), unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h4>Avg Loan Amount</h4>
                <h2>${:,.0f}</h2>
            </div>
            """.format(df['loan_amnt'].mean()), unsafe_allow_html=True)
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h4>Missing Values</h4>
                <h2>{:,}</h2>
            </div>
            """.format(df.isnull().sum().sum()), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Visualizations
        st.markdown('<h3 class="subsection-header">Factors Contributing to Default</h3>', unsafe_allow_html=True)
        
        # Default rates by categorical features
        st.plotly_chart(create_eda_visualizations(df), use_container_width=True)
        
        # Correlation heatmap
        st.plotly_chart(create_correlation_heatmap(df), use_container_width=True)
        
        # Distribution plots
        col1, col2 = st.columns(2)
        
        with col1:
            fig_age = px.histogram(df, x='person_age', color='loan_status', 
                                 title='Age Distribution by Loan Status',
                                 nbins=30)
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            fig_income = px.box(df, x='loan_status', y='person_income',
                               title='Income Distribution by Loan Status')
            st.plotly_chart(fig_income, use_container_width=True)
        
        # Key insights
        st.markdown("""
        ### Key Insights
        
        1. **Interest Rate Impact**: Higher interest rates are strongly correlated with default risk
        2. **Income vs Loan Amount**: The debt-to-income ratio is a critical factor in default prediction
        3. **Credit History**: Borrowers with previous defaults show higher likelihood of future defaults
        4. **Loan Grade**: Lower grade loans (higher risk categories) have significantly higher default rates
        5. **Employment Stability**: Longer employment history correlates with lower default risk
        """)
    
    ############# MODEL PIPELINE PAGE #############
    
    elif page == "Model Pipeline":
        st.markdown('<h2 class="section-header">Model Pipeline</h2>', unsafe_allow_html=True)
        
        # Initialize pipelines
        if 'cleaning_pipeline' not in st.session_state:
            st.session_state.cleaning_pipeline = DataCleaningPipeline()
        if 'feature_pipeline' not in st.session_state:
            st.session_state.feature_pipeline = FeatureEngineeringPipeline()
        if 'model_pipeline' not in st.session_state:
            st.session_state.model_pipeline = ModelTrainingPipeline()
        
        # =======================================================================
        # DATA CLEANING SECTION
        # =======================================================================
        
        st.markdown('<h3 class="subsection-header">Data Cleaning</h3>', unsafe_allow_html=True)
        
        if st.button("Initiate Data Cleaning", type="primary"):
            with st.spinner("Cleaning data..."):
                cleaned_data, cleaning_log = st.session_state.cleaning_pipeline.clean_data(df)
                st.session_state.cleaned_data = cleaned_data
                st.session_state.cleaning_log = cleaning_log
                st.session_state.data_cleaned = True
        
        if st.session_state.data_cleaned:
            st.success("Data cleaning completed successfully!")
            
            # Display cleaning log
            st.markdown("**Cleaning Process Log:**")
            for log_entry in st.session_state.cleaning_log:
                st.write(log_entry)
            
            # Display before/after comparison
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Original Shape", f"{df.shape[0]} × {df.shape[1]}")
            with col2:
                st.metric("Cleaned Shape", f"{st.session_state.cleaned_data.shape[0]} × {st.session_state.cleaned_data.shape[1]}")
        
        ############# DATA SPLITTING SECTION #############
        
        st.markdown("---")
        st.markdown('<h3 class="subsection-header">Data Splitting</h3>', unsafe_allow_html=True)
        
        if st.session_state.data_cleaned:
            if st.button("Split Data", type="primary"):
                with st.spinner("Splitting data..."):
                    # Prepare features and target from cleaned data
                    X = st.session_state.cleaned_data.drop(['loan_status'], axis=1)
                    y = st.session_state.cleaned_data['loan_status']
                    
                    # Split data
                    X_train, X_val, X_test, y_train, y_val, y_test = st.session_state.model_pipeline.split_data(X, y)
                    
                    st.session_state.X_train_raw = X_train
                    st.session_state.X_val_raw = X_val
                    st.session_state.X_test_raw = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_val = y_val
                    st.session_state.y_test = y_test
                    st.session_state.data_split = True
            
            if 'data_split' in st.session_state and st.session_state.data_split:
                st.success("Data split completed successfully!")
                
                # Display data split info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Training Set", f"{len(st.session_state.X_train_raw):,} ({len(st.session_state.X_train_raw)/len(st.session_state.cleaned_data):.1%})")
                with col2:
                    st.metric("Validation Set", f"{len(st.session_state.X_val_raw):,} ({len(st.session_state.X_val_raw)/len(st.session_state.cleaned_data):.1%})")
                with col3:
                    st.metric("Test Set", f"{len(st.session_state.X_test_raw):,} ({len(st.session_state.X_test_raw)/len(st.session_state.cleaned_data):.1%})")
                
                # Show default rates in each split
                st.markdown("**Default Rates by Split:**")
                split_stats = pd.DataFrame({
                    'Split': ['Training', 'Validation', 'Test'],
                    'Default Rate': [
                        f"{st.session_state.y_train.mean():.2%}",
                        f"{st.session_state.y_val.mean():.2%}",
                        f"{st.session_state.y_test.mean():.2%}"
                    ],
                    'Sample Size': [
                        len(st.session_state.y_train),
                        len(st.session_state.y_val),
                        len(st.session_state.y_test)
                    ]
                })
                st.dataframe(split_stats, use_container_width=True)
        else:
            st.warning("Please complete data cleaning first.")
        
        ############# FEATURE ENGINEERING SECTION #############
        
        st.markdown("---")
        st.markdown('<h3 class="subsection-header">Feature Engineering</h3>', unsafe_allow_html=True)
        
        if 'data_split' in st.session_state and st.session_state.data_split:
            if st.button("Generate Features", type="primary"):
                with st.spinner("Creating features..."):
                    # Fit feature engineering pipeline on training data only
                    st.session_state.feature_pipeline.fit(st.session_state.X_train_raw)
                    
                    # Transform all splits using fitted parameters
                    X_train_features, feature_descriptions = st.session_state.feature_pipeline.transform(st.session_state.X_train_raw)
                    X_val_features, _ = st.session_state.feature_pipeline.transform(st.session_state.X_val_raw)
                    X_test_features, _ = st.session_state.feature_pipeline.transform(st.session_state.X_test_raw)
                    
                    st.session_state.X_train = X_train_features
                    st.session_state.X_val = X_val_features
                    st.session_state.X_test = X_test_features
                    st.session_state.feature_descriptions = feature_descriptions
                    st.session_state.feature_names = X_train_features.columns.tolist()
                    st.session_state.features_created = True
            
            if st.session_state.features_created:
                st.success("Feature engineering completed successfully!")
                
                # Show fitted parameters used for feature engineering
                st.markdown("**Fitted Parameters (from Training Data Only):**")
                params_df = pd.DataFrame([
                    {"Parameter": "Income Quartiles (25%, 50%, 75%)", 
                     "Values": f"${st.session_state.feature_pipeline.fitted_params['income_quartiles'][0.25]:,.0f}, ${st.session_state.feature_pipeline.fitted_params['income_quartiles'][0.5]:,.0f}, ${st.session_state.feature_pipeline.fitted_params['income_quartiles'][0.75]:,.0f}"},
                    {"Parameter": "High Interest Rate Threshold (75th percentile)", 
                     "Values": f"{st.session_state.feature_pipeline.fitted_params['high_rate_threshold']:.2f}%"},
                    {"Parameter": "Debt-to-Income Median", 
                     "Values": f"{st.session_state.feature_pipeline.fitted_params['debt_to_income_median']:.3f}"},
                    {"Parameter": "Credit History Median", 
                     "Values": f"{st.session_state.feature_pipeline.fitted_params['credit_hist_median']:.1f} years"}
                ])
                st.dataframe(params_df, use_container_width=True)
                
                # Display feature table
                st.markdown("**Generated Features:**")
                feature_df = pd.DataFrame([
                    {"Feature": feat, "Description": desc} 
                    for feat, desc in st.session_state.feature_descriptions.items()
                ])
                st.dataframe(feature_df, use_container_width=True)
                
                # Show feature statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Features", len(st.session_state.feature_names))
                with col2:
                    st.metric("New Features Created", len(st.session_state.feature_descriptions))
        else:
            st.warning("Please complete data splitting first.")
        
        ############## MODEL TRAINING SECTION #############
        
        st.markdown("---")
        st.markdown('<h3 class="subsection-header">Model Training</h3>', unsafe_allow_html=True)
        
        if st.session_state.features_created:
            
            # XGBOOST MODEL TRAINING
            
            st.markdown("#### XGBoost Model")
            
            if st.button("Train XGBoost", type="primary", key="train_xgb"):
                with st.spinner("Training XGBoost model... This may take a few minutes."):
                    xgb_results = st.session_state.model_pipeline.train_xgboost(
                        st.session_state.X_train, st.session_state.X_val, st.session_state.X_test,
                        st.session_state.y_train, st.session_state.y_val, st.session_state.y_test
                    )
                    st.session_state.xgb_results = xgb_results
                    st.session_state.models_trained['xgboost'] = True
            
            if st.session_state.models_trained['xgboost']:
                st.success("XGBoost model trained successfully!")
                
                # Display results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Train AUC", f"{st.session_state.xgb_results['train_auc']:.4f}")
                with col2:
                    st.metric("Validation AUC", f"{st.session_state.xgb_results['val_auc']:.4f}")
                with col3:
                    st.metric("Test AUC", f"{st.session_state.xgb_results['test_auc']:.4f}")
                
              
                # Best parameters
                st.markdown("**Best Parameters:**")
                st.json(st.session_state.xgb_results['best_params'])
                
                # Classification report
                st.markdown("**Classification Report:**")
                st.text(st.session_state.xgb_results['classification_report'])
                
                # Feature importance
                st.markdown("**Feature Importance:**")
                fig_importance = plot_feature_importance(
                    st.session_state.xgb_results['model'], 
                    st.session_state.feature_names,
                    "XGBoost Feature Importance"
                )
                st.plotly_chart(fig_importance, use_container_width=True)
                
                # SHAP values
                st.markdown("**SHAP Values Explanation:**")
                st.info("""
                **SHAP** values show how each feature contributes to individual predictions:
                - Each dot represents one loan application
                - Features are ranked by importance (top to bottom)
                - Red dots indicate higher feature values, blue dots indicate lower values
                - Position on x-axis shows impact on prediction (right = increases default probability)
                """)
                
                try:
                    with st.spinner("Generating SHAP values..."):
                        # Sample data for SHAP (to avoid memory issues)
                        sample_size = min(500, len(st.session_state.X_test))
                        sample_indices = np.random.choice(len(st.session_state.X_test), sample_size, replace=False)
                        X_test_sample = st.session_state.X_test.iloc[sample_indices]
                        
                        shap_fig = plot_shap_values(
                            st.session_state.xgb_results['model'],
                            X_test_sample,
                            st.session_state.feature_names
                        )
                        
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            st.pyplot(shap_fig)
                except Exception as e:
                    st.warning(f"Could not generate SHAP values: {str(e)}")
            
            # LOGISTIC REGRESSION MODEL TRAINING
            
            st.markdown("#### Logistic Regression Model")
            
            if st.button("Train Logistic Regression", type="primary", key="train_lr"):
                with st.spinner("Training Logistic Regression model..."):
                    lr_results = st.session_state.model_pipeline.train_logistic_regression(
                        st.session_state.X_train, st.session_state.X_val, st.session_state.X_test,
                        st.session_state.y_train, st.session_state.y_val, st.session_state.y_test
                    )
                    st.session_state.lr_results = lr_results
                    st.session_state.models_trained['logistic'] = True
            
            if st.session_state.models_trained['logistic']:
                st.success("Logistic Regression model trained successfully!")
                
                # Display results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Train AUC", f"{st.session_state.lr_results['train_auc']:.4f}")
                with col2:
                    st.metric("Validation AUC", f"{st.session_state.lr_results['val_auc']:.4f}")
                with col3:
                    st.metric("Test AUC", f"{st.session_state.lr_results['test_auc']:.4f}")
                
                   
                # Best parameters
                st.markdown("**Best Parameters:**")
                st.json(st.session_state.lr_results['best_params'])
                
                # Classification report
                st.markdown("**Classification Report:**")
                st.text(st.session_state.lr_results['classification_report'])
                
                # Coefficients explanation
                st.markdown("**Logistic Regression Coefficients:**")
                st.info("""
                **Coefficients** in logistic regression represent the change in log-odds for a one-unit increase in the feature:
                - **Positive coefficients** (blue bars): Increase the probability of default
                - **Negative coefficients** (red bars): Decrease the probability of default
                - **Larger absolute values**: Stronger impact on the prediction
                """)
                
                # Plot coefficients
                fig_coef = plot_logistic_coefficients(
                    st.session_state.lr_results['model'],
                    st.session_state.feature_names
                )
                st.plotly_chart(fig_coef, use_container_width=True)
                
                # Interpretation table
                coeffs = st.session_state.lr_results['coefficients']
                coef_df = pd.DataFrame({
                    'Feature': st.session_state.feature_names,
                    'Coefficient': coeffs,
                    'Impact': ['Increases Default Risk' if c > 0 else 'Decreases Default Risk' for c in coeffs],
                    'Magnitude': ['Strong' if abs(c) > 0.5 else 'Moderate' if abs(c) > 0.1 else 'Weak' for c in coeffs]
                }).sort_values('Coefficient', key=abs, ascending=False)
                
                st.markdown("**Top 10 Most Influential Features:**")
                st.dataframe(coef_df.head(10), use_container_width=True)
            
            # MODEL COMPARISON
            
            if st.session_state.models_trained['xgboost'] and st.session_state.models_trained['logistic']:
                st.markdown("---")
                st.markdown("#### Model Comparison")
                
                # ROC curves comparison
                models_dict = {
                    'xgboost': st.session_state.xgb_results,
                    'logistic': st.session_state.lr_results
                }
                
                fig_roc = plot_roc_curves(models_dict, st.session_state.X_test, st.session_state.y_test)
                st.plotly_chart(fig_roc, use_container_width=True)
                
                # Comparison metrics
                comparison_df = pd.DataFrame({
                    'Model': ['XGBoost', 'Logistic Regression'],
                    'Test AUC': [st.session_state.xgb_results['test_auc'], st.session_state.lr_results['test_auc']],
                    'Validation AUC': [st.session_state.xgb_results['val_auc'], st.session_state.lr_results['val_auc']],
                    'Train AUC': [st.session_state.xgb_results['train_auc'], st.session_state.lr_results['train_auc']],
                    'Overfitting Gap': [
                        st.session_state.xgb_results['train_auc'] - st.session_state.xgb_results['val_auc'],
                        st.session_state.lr_results['train_auc'] - st.session_state.lr_results['val_auc']
                    ]
                })
                
                st.markdown("**Performance Comparison:**")
                st.dataframe(comparison_df, use_container_width=True)
        
        else:
            st.warning("Please complete feature engineering first.")
        
        ############## SUMMARY SECTION #############
        
        if st.session_state.models_trained['xgboost'] and st.session_state.models_trained['logistic']:
            st.markdown("---")
            st.markdown('<h3 class="subsection-header">Summary & Recommendations</h3>', unsafe_allow_html=True)
            
            # Performance summary
            xgb_auc = st.session_state.xgb_results['test_auc']
            lr_auc = st.session_state.lr_results['test_auc']
            
            st.markdown(f"""
            ### Model Performance Summary
            
            **XGBoost Model:**
            - Test AUC: {xgb_auc:.4f}
            - Validation AUC: {st.session_state.xgb_results['val_auc']:.4f}
            - Overfitting Gap: {st.session_state.xgb_results['train_auc'] - st.session_state.xgb_results['val_auc']:.4f}
            - Strengths: Better handling of non-linear relationships, automatic feature interactions
            
            **Logistic Regression Model:**
            - Test AUC: {lr_auc:.4f}
            - Validation AUC: {st.session_state.lr_results['val_auc']:.4f}
            - Overfitting Gap: {st.session_state.lr_results['train_auc'] - st.session_state.lr_results['val_auc']:.4f}
            - Strengths: Highly interpretable, faster training, regulatory compliance
            
            
            ### Business Recommendations
            
            **Model Selection Guidelines:**
            
            1. **For Maximum Accuracy**: Use XGBoost if performance difference is significant (>0.02 AUC)
            2. **For Regulatory Compliance**: Use Logistic Regression for explainability
            3. **For Production Deployment**: Consider ensemble of both models
            
            ### Key Risk Factors Identified
            
            Based on the properly trained models:
            - **Debt-to-Income Ratio**: Primary predictor of default risk
            - **Credit History Length**: Strong indicator of creditworthiness  
            - **Interest Rate**: Market-based risk assessment validation
            - **Employment Stability**: Important for income predictability
            - **Age Groups**: Life stage impacts financial stability
            
            """)
            
            # Final recommendation  
            if xgb_auc > lr_auc:
                winner = "XGBoost"
                margin = xgb_auc - lr_auc
            else:
                winner = "Logistic Regression"
                margin = lr_auc - xgb_auc
            
            if margin > 0.02:
                st.success(f"""
                **Final Recommendation**: {winner} shows a meaningful performance advantage ({margin:.4f} AUC difference).
                """)
            else:
                st.info(f"""
                **Final Recommendation**: Both models perform similarly ({margin:.4f} AUC difference). 
                Consider Logistic Regression for interpretability or ensemble both models.
                
                """)

############### RUN THE APP ##############

if __name__ == "__main__":
    main()
