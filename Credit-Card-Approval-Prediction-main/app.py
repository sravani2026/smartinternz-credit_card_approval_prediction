import os
import io
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file
from underwriting_engine import evaluate_applicant

app = Flask(__name__)

# Ensure the model is loaded or trained at startup
MODEL_PATH = 'best_model.joblib'

# Schema check to clear old models
if os.path.exists(MODEL_PATH):
    try:
        temp_model = joblib.load(MODEL_PATH)
        if hasattr(temp_model, 'named_steps') and 'preprocessor' in temp_model.named_steps:
            preprocessor = temp_model.named_steps['preprocessor']
            transformers = preprocessor.transformers
            for name, trans, cols in transformers:
                if 'Credit_Score' in cols:
                    print("Detected old schema in best_model.joblib. Removing old model file to force retraining...")
                    os.remove(MODEL_PATH)
                    break
    except Exception as e:
        print(f"Warning checking model schema: {e}")

if not os.path.exists(MODEL_PATH):
    print(f"Model not found at {MODEL_PATH}. Training model dynamically with Indian parameters...")
    try:
        import ml_pipeline
        df = ml_pipeline.generate_synthetic_data(n_samples=1500)
        X, y = ml_pipeline.preprocess_data(df)
        best_model, _, _ = ml_pipeline.train_and_evaluate_models(X, y)
        joblib.dump(best_model, MODEL_PATH)
        print("Model trained and saved successfully.")
    except Exception as e:
        print(f"Error training model at startup: {e}")
        raise e

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Page Routes
@app.route('/')
@app.route('/customer')
def customer_dashboard():
    """
    Renders the Customer Application Portal.
    """
    return render_template('customer.html')

@app.route('/analyst')
def analyst_dashboard():
    """
    Renders the Analyst Back-Office & Batch Processing Dashboard.
    """
    return render_template('analyst.html')

@app.route('/compliance')
def compliance_dashboard():
    """
    Renders the Compliance & Audit Log Interface.
    """
    return render_template('compliance.html')

# API Endpoints
@app.route('/predict_single', methods=['POST'])
def predict_single():
    """
    Accepts JSON input data from the frontend form for individual credit screening.
    Expected keys: Age, Annual_Income, CIBIL_Score, Debt, Years_Employed, Prior_Default, override (optional)
    """
    if not model:
        return jsonify({'error': 'Model not loaded on server'}), 500
        
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
            
        required_fields = ['Age', 'Annual_Income', 'CIBIL_Score', 'Debt', 'Years_Employed', 'Prior_Default']
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return jsonify({'error': f'Missing fields in request payload: {missing_fields}'}), 400
            
        # Parse inputs
        age = float(data['Age'])
        annual_income = float(data['Annual_Income'])
        cibil_score = float(data['CIBIL_Score'])
        debt = float(data['Debt'])
        years_employed = float(data['Years_Employed'])
        prior_default = str(data['Prior_Default']).strip()
        override = bool(data.get('override', False))
        
        if prior_default not in ['Yes', 'No']:
            return jsonify({'error': "Prior_Default must be 'Yes' or 'No'"}), 400
            
        # Evaluate using Central Underwriting Decision Engine (Source of Truth)
        decision = evaluate_applicant(
            age=age,
            annual_income=annual_income,
            cibil_score=cibil_score,
            debt=debt,
            years_employed=years_employed,
            prior_default=prior_default,
            model=model,
            override_authorized=override
        )
        
        response = {
            'risk_score': decision['risk_score'],
            'risk_classification': decision['risk_classification'],
            'approval_status': decision['approval_status'],
            'rejection_reasons': decision['rejection_reasons'],
            'all_explanations': decision['all_explanations'],
            'triggered_rules': decision['triggered_rules'],
            'overridden': decision['overridden'],
            'input_data': data
        }
        return jsonify(response), 200
        
    except ValueError as ve:
        return jsonify({'error': f'Invalid value type: {str(ve)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction execution failed: {str(e)}'}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Accepts a CSV file upload, runs batch classification using central underwriting engine,
    and returns the CSV with Risk_Score, Risk_Classification, and Approval_Status appended.
    """
    if not model:
        return jsonify({'error': 'Model not loaded on server'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request (use parameter name "file")'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and file.filename.endswith('.csv'):
        try:
            df = pd.read_csv(file)
            
            # Check for required feature columns
            required_cols = ['Age', 'Annual_Income', 'CIBIL_Score', 'Debt', 'Years_Employed', 'Prior_Default']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return jsonify({'error': f'CSV is missing required feature columns: {missing_cols}'}), 400
                
            # Process row-by-row through the central underwriting engine
            risk_scores = []
            risk_classifications = []
            approval_statuses = []
            rejection_reasons_list = []
            triggered_rules_list = []
            overridden_list = []
            
            for idx, row in df.iterrows():
                decision = evaluate_applicant(
                    age=float(row['Age']),
                    annual_income=float(row['Annual_Income']),
                    cibil_score=int(row['CIBIL_Score']),
                    debt=float(row['Debt']),
                    years_employed=float(row['Years_Employed']),
                    prior_default=str(row['Prior_Default']).strip(),
                    model=model,
                    override_authorized=False
                )
                risk_scores.append(decision['risk_score'])
                risk_classifications.append(decision['risk_classification'])
                approval_statuses.append(decision['approval_status'])
                rejection_reasons_list.append("; ".join(decision['all_explanations']))
                triggered_rules_list.append("; ".join(decision['triggered_rules']))
                overridden_list.append(decision['overridden'])
                
            # Append decision fields to output dataframe copy
            df['Risk_Score'] = risk_scores
            df['Risk_Classification'] = risk_classifications
            df['Approval_Status'] = approval_statuses
            df['Rejection_Reasons'] = rejection_reasons_list
            df['Triggered_Rules'] = triggered_rules_list
            df['Overridden'] = overridden_list
            
            # Save results back to bytes buffer
            output = io.BytesIO()
            df.to_csv(output, index=False)
            output.seek(0)
            
            return send_file(
                output,
                mimetype='text/csv',
                as_attachment=True,
                download_name='credit_risk_predictions.csv'
            )
            
        except Exception as e:
            return jsonify({'error': f'Batch processing failed: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Invalid file format. Please upload a valid CSV file.'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
