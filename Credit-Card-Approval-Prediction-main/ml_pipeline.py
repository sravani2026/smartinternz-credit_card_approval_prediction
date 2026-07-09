import os
import joblib
import pandas as pd
import numpy as np

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# WML Client
try:
    from ibm_watson_machine_learning import APIClient
except ImportError:
    try:
        from ibm_watsonx_ai import APIClient
    except ImportError:
        APIClient = None


def generate_synthetic_data(n_samples=1000, random_state=42):
    """
    Generates realistic synthetic credit card applicant data based on the Indian context.
    - Currency: Indian Rupees (₹ - INR)
    - Credit Bureau Rating: CIBIL Score (300 to 900)
    """
    np.random.seed(random_state)
    
    # Demographics & Financials (INR values represent Annual Lakhs Per Annum - LPA scales)
    age = np.random.randint(18, 75, size=n_samples)
    annual_income = np.random.exponential(scale=450000, size=n_samples) + 180000
    cibil_score = np.random.randint(300, 900, size=n_samples)
    debt = annual_income * np.random.uniform(0.0, 0.55, size=n_samples)
    years_employed = np.random.uniform(0, 40, size=n_samples)
    
    prior_default = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.10, 0.90])
    
    # Payment status codes: C (closed), X (no activity), 0 (1-29 days), 1 (30-59 days), 
    # 2 (60-89 days), 3 (90-119 days), 4 (120-149 days), 5 (150+ days/default)
    payment_status = []
    for i in range(n_samples):
        # Risk score calculation calibrated for CIBIL & INR
        risk_score = 0.0
        if prior_default[i] == 'Yes':
            risk_score += 3.0
        if cibil_score[i] < 650:
            risk_score += 2.0
        if debt[i] / annual_income[i] > 0.45:
            risk_score += 1.5
        if age[i] < 25:
            risk_score += 0.5
        
        # Add noise
        risk_score += np.random.normal(0, 1.0)
        
        if risk_score > 4.5:
            status = np.random.choice(['2', '3', '4', '5'], p=[0.25, 0.25, 0.25, 0.25])
        elif risk_score > 2.0:
            status = np.random.choice(['0', '1', '2'], p=[0.6, 0.3, 0.1])
        else:
            status = np.random.choice(['C', 'X', '0'], p=[0.5, 0.4, 0.1])
            
        payment_status.append(status)
        
    df = pd.DataFrame({
        'Applicant_ID': [f'APP_{1000 + i}' for i in range(n_samples)],
        'Age': age,
        'Annual_Income': np.round(annual_income, 0),
        'CIBIL_Score': cibil_score,
        'Debt': np.round(debt, 0),
        'Years_Employed': np.round(years_employed, 1),
        'Prior_Default': prior_default,
        'Payment_Status': payment_status
    })
    
    return df


def preprocess_data(df):
    """
    Transforms multi-class past-due payment status codes into binary labels:
    0 = Eligible (low risk)
    1 = High-Risk (default probability)
    """
    eligible_codes = ['C', 'X', '0', '1', 0, 1]
    
    # Explicit conversion of payment status codes into binary labels
    y = df['Payment_Status'].apply(lambda x: 0 if str(x).strip().upper() in [str(c) for c in eligible_codes] else 1)
    
    # Extract features (exclude target identifier and multi-class source labels)
    X = df.drop(columns=['Applicant_ID', 'Payment_Status'], errors='ignore')
    
    return X, y


def train_and_evaluate_models(X, y):
    """
    Trains and evaluates Logistic Regression, Random Forest, XGBoost, and Decision Tree.
    Returns the best model pipeline.
    """
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define preprocessing pipeline for features
    numeric_features = ['Age', 'Annual_Income', 'CIBIL_Score', 'Debt', 'Years_Employed']
    categorical_features = ['Prior_Default']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='if_binary'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=6),
        'XGBoost': XGBClassifier(random_state=42, n_estimators=100, max_depth=4, eval_metric='logloss')
    }
    
    results = {}
    best_f1 = -1
    best_model_name = None
    best_pipeline = None
    
    for name, model in models.items():
        # Create full pipeline
        clf_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Fit
        clf_pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = clf_pipeline.predict(X_test)
        
        # Compute metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            y_prob = clf_pipeline.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            auc = 0.5  # fallback
            
        results[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc,
            'pipeline': clf_pipeline
        }
        
        print(f"--- {name} ---")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-Score : {f1:.4f}")
        print(f"ROC-AUC  : {auc:.4f}\n")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_pipeline = clf_pipeline
            
    print(f"Best performing model: {best_model_name} with F1-Score {best_f1:.4f}")
    return best_pipeline, best_model_name, results


def deploy_to_ibm_wml(model_pipeline, api_key, url, space_id, model_name="Credit_Card_Approval_Model_INR", deployment_name="Credit_Card_Approval_Deployment_INR"):
    """
    Authenticates, saves, and deploys the model pipeline to IBM Watson Machine Learning.
    """
    if APIClient is None:
        raise ImportError(
            "IBM Watson Machine Learning SDK is not installed."
        )
        
    print("Initiating IBM Watson Machine Learning authentication...")
    wml_credentials = {
        "apikey": api_key,
        "url": url
    }
    
    client = APIClient(wml_credentials)
    client.set.default_space(space_id)
    
    print("Fetching software specifications...")
    spec_name = "runtime-23.1-py3.10"
    try:
        software_spec_uid = client.software_specifications.get_id_by_name(spec_name)
    except Exception as e:
        print(f"Warning: Could not get software spec ID for '{spec_name}': {e}")
        software_spec_uid = client.software_specifications.get_id_by_name("default_py3.10")
            
    print(f"Using software specification UID: {software_spec_uid}")
    
    model_meta_props = {
        client.repository.ModelMetaNames.NAME: model_name,
        client.repository.ModelMetaNames.TYPE: "scikit-learn_1.3",
        client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: software_spec_uid
    }
    
    published_model = client.repository.store_model(
        model=model_pipeline,
        meta_props=model_meta_props
    )
    
    model_uid = client.repository.get_model_id(published_model)
    print(f"Model stored successfully. Model UID: {model_uid}")
    
    print(f"Creating online deployment '{deployment_name}'...")
    deployment_meta_props = {
        client.deployments.ConfigurationMetaNames.NAME: deployment_name,
        client.deployments.ConfigurationMetaNames.ONLINE: {}
    }
    
    deployment = client.deployments.create(
        artifact_uid=model_uid,
        meta_props=deployment_meta_props
    )
    
    deployment_uid = client.deployments.get_id(deployment)
    scoring_href = client.deployments.get_scoring_href(deployment)
    
    return {
        "model_uid": model_uid,
        "deployment_uid": deployment_uid,
        "scoring_endpoint": scoring_href
    }


if __name__ == "__main__":
    print("=== Indian Credit Card Approval ML Pipeline ===")
    
    data_path = 'credit_data.csv'
    
    # Schema check to clear old schema data
    if os.path.exists(data_path):
        try:
            test_df = pd.read_csv(data_path)
            if 'Credit_Score' in test_df.columns:
                print("Old US schema 'Credit_Score' found in credit_data.csv. Removing file to regenerate Indian parameters...")
                os.remove(data_path)
        except Exception as e:
            print(f"Warning checking schema: {e}")
            
    if os.path.exists(data_path):
        print(f"Loading dataset from: {data_path}")
        df = pd.read_csv(data_path)
    else:
        print(f"Dataset not found at {data_path}. Generating synthetic Indian applicant credit data...")
        df = generate_synthetic_data(n_samples=1500)
        df.to_csv(data_path, index=False)
        print(f"Synthetic Indian dataset saved to: {data_path}")
        
    print("Preprocessing data and converting payment status codes to binary labels...")
    X, y = preprocess_data(df)
    print(f"Preprocessed features shape: {X.shape}, labels distribution:\n{y.value_counts(normalize=True)}")
    
    print("\nTraining and evaluating model candidates...")
    best_model, best_name, evaluation_results = train_and_evaluate_models(X, y)
    
    local_model_path = 'best_model.joblib'
    print(f"Saving best model ({best_name}) locally to: {local_model_path}")
    joblib.dump(best_model, local_model_path)
    print("Model saved successfully.")
    
    wml_api_key = os.getenv("WML_API_KEY")
    wml_url = os.getenv("WML_URL")
    wml_space_id = os.getenv("WML_SPACE_ID")
    
    if wml_api_key and wml_url and wml_space_id:
        print("\nWatson Machine Learning credentials found. Starting deployment...")
        try:
            deployment_details = deploy_to_ibm_wml(
                model_pipeline=best_model,
                api_key=wml_api_key,
                url=wml_url,
                space_id=wml_space_id
            )
            print("WML Deployment Details:", deployment_details)
        except Exception as e:
            print(f"Error during WML deployment: {e}")
    else:
        print("\nNote: IBM WML environment variables not set. Skipping deployment.")
