# Employee Attrition Prediction - Notebook Implementation Plan

## Dataset Analysis

From the GitHub repository, the employee.csv contains the following columns:

### Available Columns:
- **EMPLOYEE_CODE** - Unique identifier (PII - exclude from modeling)
- **FIRST_NAME** - Employee first name (PII - exclude)
- **FIRST_NAME_MB** - First name multibyte (PII - exclude)
- **LAST_NAME** - Employee last name (PII - exclude)
- **LAST_NAME_MB** - Last name multibyte (PII - exclude)
- **DATE_HIRED** - Hiring date (use for tenure calculation)
- **TERMINATION_DATE** - Termination date (use for target variable)
- **TERMINATION_CODE** - Termination reason code
- **BIRTH_DATE** - Date of birth (use for age calculation)
- **GENDER_CODE** - Gender (fairness monitoring feature)
- **WORK_PHONE** - Work phone (PII - exclude)
- **EXTENSION** - Phone extension (PII - exclude)
- **FAX** - Fax number (PII - exclude)
- **EMAIL** - Email address (PII - exclude)
- **SSN** - Social Security Number (PII - MUST exclude)
- **COMMUTE_TIME** - Commute time in minutes (predictive feature)

---

## Notebook Structure (12 Sections)

### 1. Business Objective & Use Case
**Content:**
- Executive summary of employee attrition problem
- Business impact (cost of turnover, productivity loss)
- ML solution value proposition
- Success metrics definition

### 2. Environment Setup (watsonx.ai / WML SDK)
**Content:**
- Import required libraries (pandas, numpy, sklearn, ibm_watson_machine_learning)
- Set up environment variables for credentials (NO hardcoding)
- Initialize Watson Machine Learning client
- Verify connectivity to watsonx.ai

**Security Best Practices:**
```python
# Use environment variables
WML_API_KEY = os.getenv('WML_API_KEY')
WML_URL = os.getenv('WML_URL', 'https://us-south.ml.cloud.ibm.com')
SPACE_ID = os.getenv('SPACE_ID')
```

### 3. Data Loading & Initial Exploration
**Content:**
- Load employee.csv from GitHub URL
- Display dataset shape and info
- Show sample records
- Identify data types
- Check for missing values
- Display basic statistics

### 4. Data Cleaning & Feature Engineering
**Content:**

**PII Exclusion (Explicit):**
- Drop columns: EMPLOYEE_CODE, FIRST_NAME, FIRST_NAME_MB, LAST_NAME, LAST_NAME_MB, WORK_PHONE, EXTENSION, FAX, EMAIL, SSN
- Document why each is excluded (privacy, GDPR compliance)

**Feature Engineering:**
1. **AGE** - Calculate from BIRTH_DATE
   ```python
   df['AGE'] = (pd.Timestamp.now() - pd.to_datetime(df['BIRTH_DATE'])).dt.days / 365.25
   ```

2. **TENURE_YEARS** - Calculate from DATE_HIRED
   ```python
   df['TENURE_YEARS'] = (pd.Timestamp.now() - pd.to_datetime(df['DATE_HIRED'])).dt.days / 365.25
   ```

3. **TENURE_MONTHS** - More granular tenure
   ```python
   df['TENURE_MONTHS'] = (pd.Timestamp.now() - pd.to_datetime(df['DATE_HIRED'])).dt.days / 30.44
   ```

4. **COMMUTE_TIME** - Already available, handle missing values

5. **GENDER_CODE** - Encode as categorical (note fairness implications)
   ```python
   df['GENDER_ENCODED'] = LabelEncoder().fit_transform(df['GENDER_CODE'].fillna('Unknown'))
   ```

6. **AGE_GROUP** - Categorical age bands for interpretability
   ```python
   df['AGE_GROUP'] = pd.cut(df['AGE'], bins=[0, 25, 35, 45, 55, 100], 
                             labels=['<25', '25-35', '35-45', '45-55', '55+'])
   ```

7. **TENURE_GROUP** - Categorical tenure bands
   ```python
   df['TENURE_GROUP'] = pd.cut(df['TENURE_YEARS'], bins=[0, 1, 3, 5, 10, 100],
                                labels=['<1yr', '1-3yr', '3-5yr', '5-10yr', '10+yr'])
   ```

**Missing Value Strategy:**
- COMMUTE_TIME: Impute with median
- GENDER_CODE: Create 'Unknown' category
- Document all imputation decisions

### 5. Target Variable Definition
**Content:**
```python
# Binary target: 1 if terminated, 0 if active
df['ATTRITION'] = df['TERMINATION_DATE'].notna().astype(int)

# Display class distribution
print(f"Attrition Rate: {df['ATTRITION'].mean():.2%}")
print(f"Active Employees: {(df['ATTRITION']==0).sum()}")
print(f"Terminated Employees: {(df['ATTRITION']==1).sum()}")
```

**Handle Class Imbalance:**
- Check if dataset is imbalanced
- Consider SMOTE or class weights if needed
- Document strategy

### 6. Train/Test Split
**Content:**
```python
# Feature selection
feature_cols = ['AGE', 'TENURE_YEARS', 'TENURE_MONTHS', 'COMMUTE_TIME', 'GENDER_ENCODED']
X = df[feature_cols]
y = df['ATTRITION']

# Stratified split to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Training attrition rate: {y_train.mean():.2%}")
print(f"Test attrition rate: {y_test.mean():.2%}")
```

### 7. Model Training
**Content:**

**Model Selection Rationale:**

**Selected Model: Random Forest Classifier** ✅

**Why Random Forest for Employee Attrition:**
- **Handles Non-Linear Relationships:** Employee attrition is influenced by complex interactions between features (e.g., age + tenure + commute time)
- **Feature Importance Built-In:** Provides clear rankings of which factors drive attrition
- **Robust Performance:** Works well out-of-box without extensive tuning
- **Handles Missing Data:** Naturally robust to missing values
- **No Feature Scaling Required:** Works with raw features
- **Ensemble Approach:** Reduces overfitting through averaging multiple decision trees

**Governance Considerations:**
- **Explainability:** Use feature importance + SHAP/LIME for individual predictions
- **Fairness Monitoring:** OpenScale can monitor for bias regardless of model type
- **Interpretability:** While less transparent than logistic regression, feature importance and tree visualization provide insights
- **Regulatory Compliance:** Combined with OpenScale explainability, meets governance requirements

**Alternative Considered:**
- **Logistic Regression:** More interpretable but may miss complex patterns in employee behavior

**Implementation:**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Create pipeline (scaling optional for Random Forest but included for consistency)
model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        class_weight='balanced',  # Handle class imbalance
        n_jobs=-1  # Use all CPU cores
    ))
])

# Train model
model_pipeline.fit(X_train, y_train)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model_pipeline.named_steps['classifier'].feature_importances_
}).sort_values('importance', ascending=False)

print("Feature Importance:")
print(feature_importance)

# Visualize feature importance
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.show()
```

### 8. Model Evaluation
**Content:**
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)

# Predictions
y_pred = model_pipeline.predict(X_test)
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

# Metrics
metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1-Score': f1_score(y_test, y_pred),
    'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
}

print("Model Performance Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

**Visualization:**
- ROC curve
- Confusion matrix heatmap
- Feature importance chart

### 9. Model Registration in Watson Machine Learning
**Content:**
```python
import joblib
from ibm_watson_machine_learning import APIClient

# Initialize WML client
wml_credentials = {
    "url": os.getenv('WML_URL'),
    "apikey": os.getenv('WML_API_KEY')
}

client = APIClient(wml_credentials)
client.set.default_space(os.getenv('SPACE_ID'))

# Save model locally
model_filename = 'employee_attrition_model.pkl'
joblib.dump(model_pipeline, model_filename)

# Model metadata
model_metadata = {
    client.repository.ModelMetaNames.NAME: "EmployeeAttritionModel",
    client.repository.ModelMetaNames.TYPE: "scikit-learn_1.3",
    client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: client.software_specifications.get_id_by_name("runtime-23.1-py3.10"),
    client.repository.ModelMetaNames.LABEL_FIELD: "ATTRITION",
    client.repository.ModelMetaNames.TRAINING_DATA_REFERENCES: [{
        "type": "fs",
        "location": {
            "path": "employee.csv"
        },
        "schema": {
            "id": "1",
            "fields": [
                {"name": col, "type": "double"} for col in feature_cols
            ] + [{"name": "ATTRITION", "type": "integer"}]
        }
    }]
}

# Store model
stored_model = client.repository.store_model(
    model=model_filename,
    meta_props=model_metadata
)

model_uid = client.repository.get_model_id(stored_model)
print(f"Model stored with UID: {model_uid}")
```

### 10. Model Deployment to watsonx.ai
**Content:**
```python
# Deployment metadata
deployment_metadata = {
    client.deployments.ConfigurationMetaNames.NAME: "EmployeeAttritionDeployment",
    client.deployments.ConfigurationMetaNames.ONLINE: {},
    client.deployments.ConfigurationMetaNames.HARDWARE_SPEC: {
        "name": "S",
        "num_nodes": 1
    }
}

# Deploy model
deployment = client.deployments.create(
    model_uid,
    meta_props=deployment_metadata
)

deployment_uid = client.deployments.get_uid(deployment)
print(f"Deployment UID: {deployment_uid}")

# Get scoring endpoint
scoring_endpoint = client.deployments.get_scoring_href(deployment)
print(f"Scoring Endpoint: {scoring_endpoint}")
```

### 11. Test Inference (Prediction API)
**Content:**
```python
# Create test payload
test_payload = {
    "input_data": [{
        "fields": feature_cols,
        "values": [
            [35, 5.2, 62, 45, 1],  # Example: 35 years old, 5.2 years tenure, 45 min commute
            [28, 1.5, 18, 30, 0],  # Example: 28 years old, 1.5 years tenure, 30 min commute
            [52, 15.0, 180, 60, 1] # Example: 52 years old, 15 years tenure, 60 min commute
        ]
    }]
}

# Score model
predictions = client.deployments.score(deployment_uid, test_payload)

print("Predictions:")
print(json.dumps(predictions, indent=2))

# Interpret results
for i, pred in enumerate(predictions['predictions'][0]['values']):
    attrition_prob = pred[1]  # Probability of attrition
    risk_level = "HIGH" if attrition_prob > 0.7 else "MEDIUM" if attrition_prob > 0.4 else "LOW"
    print(f"\nEmployee {i+1}:")
    print(f"  Attrition Probability: {attrition_prob:.2%}")
    print(f"  Risk Level: {risk_level}")
```

### 12. Next Steps: Governance & Monitoring (OpenScale)
**Content:**

**Explain OpenScale Integration (No Implementation):**

#### A. Bias Monitoring
```markdown
**Purpose:** Detect and mitigate bias in predictions across protected attributes

**Configuration for Employee Attrition:**
- **Protected Attribute:** GENDER_CODE
- **Favorable Outcome:** ATTRITION = 0 (employee stays)
- **Unfavorable Outcome:** ATTRITION = 1 (employee leaves)
- **Fairness Threshold:** 95% (disparate impact ratio)

**Monitoring:**
- Track prediction disparities between male/female employees
- Alert if model predicts higher attrition for one gender
- Ensure compliance with EEOC guidelines

**Example Alert:**
"Model predicts 15% higher attrition rate for female employees 
compared to male employees with similar profiles. 
Fairness threshold violated (85% < 95%)."
```

#### B. Drift Detection
```markdown
**Data Drift:**
- Monitor changes in input feature distributions
- Track: AGE, TENURE_YEARS, COMMUTE_TIME distributions
- Alert if current data differs significantly from training data

**Model Drift:**
- Monitor prediction accuracy over time
- Track: Accuracy, Precision, Recall trends
- Alert if performance degrades below thresholds

**Example Scenarios:**
1. Company relocates → COMMUTE_TIME distribution shifts
2. Hiring surge → AGE distribution changes
3. Economic downturn → Attrition patterns change
```

#### C. Explainability
```markdown
**LIME (Local Interpretable Model-agnostic Explanations):**
- Explain individual predictions
- Show which features contributed most to prediction

**Example Explanation:**
"Employee predicted to leave (78% probability) because:
- High commute time (60 min) → +25% risk
- Low tenure (1.5 years) → +20% risk
- Young age (28) → +15% risk"

**Use Cases:**
- HR can take targeted retention actions
- Employees understand factors affecting predictions
- Regulatory compliance (right to explanation)
```

#### D. Integration Steps
```markdown
1. **Connect OpenScale to WML:**
   - Authenticate with IBM Cloud API key
   - Link to deployment space
   - Subscribe to deployed model

2. **Configure Monitors:**
   - Quality Monitor: Set accuracy threshold (85%)
   - Fairness Monitor: Configure gender as protected attribute
   - Drift Monitor: Enable data and model drift detection
   - Explainability: Enable LIME explanations

3. **Set Up Feedback Loop:**
   - Collect actual attrition outcomes
   - Feed back to OpenScale for accuracy tracking
   - Retrain model when drift detected

4. **Create Dashboards:**
   - Real-time fairness metrics
   - Model performance trends
   - Drift alerts and notifications
   - Explainability reports
```

#### E. Governance Best Practices
```markdown
**Model Documentation:**
- Model card with intended use, limitations, training data
- Feature importance documentation
- Bias testing results
- Performance benchmarks

**Monitoring Schedule:**
- Daily: Check for new predictions and alerts
- Weekly: Review fairness and drift metrics
- Monthly: Generate governance reports
- Quarterly: Model retraining evaluation

**Stakeholder Communication:**
- HR: Actionable insights for retention
- Legal: Compliance documentation
- Executives: Business impact metrics
- Data Science: Model performance tracking
```

---

## Implementation Checklist

### Security & Best Practices
- ✅ Use environment variables for all credentials
- ✅ Never hardcode API keys or passwords
- ✅ Explicitly exclude all PII columns
- ✅ Document data privacy decisions
- ✅ Use joblib for sklearn model serialization
- ✅ Implement proper error handling
- ✅ Add logging for debugging

### Code Quality
- ✅ Clear markdown documentation for each section
- ✅ Well-commented Python code
- ✅ Consistent naming conventions
- ✅ Modular, reusable functions
- ✅ Type hints where appropriate

### Enterprise Readiness
- ✅ Production-quality error handling
- ✅ Comprehensive logging
- ✅ Model versioning strategy
- ✅ Deployment rollback plan
- ✅ Monitoring and alerting setup
- ✅ Documentation for operations team

### Governance & Compliance
- ✅ Fairness considerations documented
- ✅ Bias monitoring strategy defined
- ✅ Explainability approach outlined
- ✅ Regulatory compliance notes
- ✅ Model card template

---

## Expected Notebook Outputs

1. **Data Insights:**
   - Dataset shape and statistics
   - Attrition rate and distribution
   - Feature correlations

2. **Model Performance:**
   - Accuracy: ~85-90%
   - ROC-AUC: ~0.85-0.92
   - Precision/Recall: Balanced for business needs

3. **Deployment Artifacts:**
   - Model UID in WML
   - Deployment UID
   - Scoring endpoint URL
   - Sample predictions

4. **Governance Documentation:**
   - Feature importance rankings
   - Fairness monitoring plan
   - Drift detection strategy
   - Explainability examples

---

## Next Steps After Notebook Creation

1. **Test in Watson Studio:**
   - Import notebook to Watson Studio
   - Run all cells sequentially
   - Verify outputs

2. **Validate Deployment:**
   - Test scoring endpoint
   - Verify response format
   - Check latency

3. **Set Up OpenScale:**
   - Follow integration steps in Section 12
   - Configure all monitors
   - Test alerting

4. **Production Handoff:**
   - Document deployment process
   - Train operations team
   - Establish monitoring schedule
   - Create runbook for issues

---

**Ready to implement!** This plan provides a complete blueprint for creating a production-quality Employee Attrition Prediction notebook with IBM watsonx.ai integration.