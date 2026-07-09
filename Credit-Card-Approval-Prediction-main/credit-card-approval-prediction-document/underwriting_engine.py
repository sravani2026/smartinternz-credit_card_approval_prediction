import pandas as pd

def evaluate_applicant(age, annual_income, cibil_score, debt, years_employed, prior_default, model=None, override_authorized=False):
    """
    Centralized Credit Card Underwriting Engine (Source of Truth).
    Returns a binary decision:
      - Approved
      - Rejected
      
    Derived Metric:
    DTI = (debt / annual_income * 100) if annual_income > 0 else 100.0
    
    Hard Rejection Rules (Decision = Rejected):
    1. CIBIL < 550
    2. Prior Default History = Yes
    3. Age < 18
    4. DTI > 60
    5. Annual Income < 60,000
    """
    rejection_reasons = []
    triggered_rules = []
    
    # Calculate derived metric
    dti = (debt / annual_income * 100) if annual_income > 0 else 100.0
    
    # Evaluate Hard Rejection Rules
    if cibil_score < 550:
        rejection_reasons.append(f"CIBIL Score below threshold ({cibil_score} < 550)")
        triggered_rules.append("RULE_CIBIL_MIN")
        
    if prior_default == 'Yes':
        rejection_reasons.append("Prior credit default history reported")
        triggered_rules.append("RULE_PRIOR_DEFAULT")
        
    if age < 18:
        rejection_reasons.append(f"Applicant is underage ({age} < 18)")
        triggered_rules.append("RULE_AGE_MIN")
        
    if dti > 60:
        rejection_reasons.append(f"Debt-to-Income (DTI) ratio exceeds extreme limit ({dti:.1f}% > 60%)")
        triggered_rules.append("RULE_DTI_MAX")
        
    if annual_income < 60000:
        rejection_reasons.append(f"Annual Income below policy limit (₹{annual_income:,.2f} < ₹60,000)")
        triggered_rules.append("RULE_MIN_INCOME")
        
    has_hard_fail = len(rejection_reasons) > 0
    
    # ML Scoring
    ml_risk_score = 0.5
    ml_prediction = 0
    
    if model:
        try:
            input_df = pd.DataFrame([{
                'Age': age,
                'Annual_Income': annual_income,
                'CIBIL_Score': cibil_score,
                'Debt': debt,
                'Years_Employed': years_employed,
                'Prior_Default': prior_default
            }])
            ml_prediction = int(model.predict(input_df)[0])
            ml_risk_score = float(model.predict_proba(input_df)[0][1])
        except Exception as e:
            print(f"ML Scoring Error: {e}")
            
    # Resolve Decision
    overridden = False
    
    if has_hard_fail:
        if override_authorized:
            approval_status = "Approved"
            risk_classification = "High-Risk"
            overridden = True
        else:
            approval_status = "Rejected"
            risk_classification = "High-Risk"
            ml_risk_score = max(ml_risk_score, 0.90)  # scale risk score to high
    else:
        # Standard flow (use ML model prediction)
        if ml_prediction == 1 or ml_risk_score > 0.5:
            approval_status = "Rejected"
            risk_classification = "High-Risk"
        else:
            approval_status = "Approved"
            risk_classification = "Eligible"
            
    return {
        "approval_status": approval_status,
        "risk_classification": risk_classification,
        "risk_score": round(ml_risk_score, 4),
        "rejection_reasons": rejection_reasons,
        "triggered_rules": triggered_rules,
        "all_explanations": rejection_reasons,
        "overridden": overridden
    }
