import unittest
from underwriting_engine import evaluate_applicant

class TestBinaryUnderwritingEngine(unittest.TestCase):
    
    def test_rejection_cibil(self):
        # CIBIL < 550
        res = evaluate_applicant(age=30, annual_income=500000, cibil_score=549, debt=50000, years_employed=5, prior_default='No')
        self.assertEqual(res['approval_status'], 'Rejected')
        self.assertIn('RULE_CIBIL_MIN', res['triggered_rules'])
        
    def test_rejection_default(self):
        # Prior Default = Yes
        res = evaluate_applicant(age=30, annual_income=500000, cibil_score=750, debt=50000, years_employed=5, prior_default='Yes')
        self.assertEqual(res['approval_status'], 'Rejected')
        self.assertIn('RULE_PRIOR_DEFAULT', res['triggered_rules'])
        
    def test_rejection_age(self):
        # Age < 18
        res = evaluate_applicant(age=17, annual_income=500000, cibil_score=750, debt=50000, years_employed=5, prior_default='No')
        self.assertEqual(res['approval_status'], 'Rejected')
        self.assertIn('RULE_AGE_MIN', res['triggered_rules'])
        
    def test_rejection_dti(self):
        # DTI > 60
        res = evaluate_applicant(age=30, annual_income=500000, cibil_score=750, debt=350000, years_employed=5, prior_default='No') # DTI = 70%
        self.assertEqual(res['approval_status'], 'Rejected')
        self.assertIn('RULE_DTI_MAX', res['triggered_rules'])
        
    def test_rejection_income(self):
        # Annual Income < 60,000
        res = evaluate_applicant(age=30, annual_income=59000, cibil_score=750, debt=10000, years_employed=5, prior_default='No')
        self.assertEqual(res['approval_status'], 'Rejected')
        self.assertIn('RULE_MIN_INCOME', res['triggered_rules'])

    def test_eligible_profile(self):
        # All rules pass
        res = evaluate_applicant(age=30, annual_income=150000, cibil_score=750, debt=30000, years_employed=5, prior_default='No', model=None)
        self.assertEqual(res['approval_status'], 'Approved')
        self.assertEqual(res['risk_classification'], 'Eligible')
        self.assertEqual(len(res['triggered_rules']), 0)
        
    def test_rejection_override(self):
        # Rejected profile with override
        res = evaluate_applicant(age=30, annual_income=500000, cibil_score=400, debt=50000, years_employed=5, prior_default='No', override_authorized=True)
        self.assertEqual(res['approval_status'], 'Approved')
        self.assertEqual(res['risk_classification'], 'High-Risk')
        self.assertTrue(res['overridden'])

if __name__ == '__main__':
    unittest.main()
