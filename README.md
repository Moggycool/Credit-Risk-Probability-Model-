# Credit-Risk-Probability-Model-
An End-to-End Implementation for Building, Deploying, and Automating a Credit Risk Model
## Credit Scoring Business Understanding

### How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?
The Basel II Capital Accord is an international banking regulation that emphasizes rigorous risk measurement and management practices, requiring banks to calculate adequate capital based on their portfolio risk—including credit risk—using standardized and internal model-based approaches. Basel II highlights the need for transparency, interpretability, and auditability in all risk models used for regulatory reporting and capital allocation.

For Bati Bank, this means any credit scoring model must be thoroughly documented and interpretable by internal risk teams, auditors, and regulators. Complex, opaque "black-box" models may lead to compliance issues if their outputs can’t be explained or justified based on input data and logic. An interpretable model ensures:
- Transparency in how risk scores are produced
- The ability to explain loan decisions to customers and regulators
- Easier identification and correction of model bias or errors
- Streamlined validation and audit processes

### Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?
In the provided e-commerce dataset, we do not have an explicit "default" outcome (i.e., whether a user actually defaulted on a loan, since this is a new service). To train a supervised model, we must engineer a proxy variable that acts as a stand-in for future default risk—such as labeling disengaged or low-activity customers as “high risk” based on their Recency, Frequency, and Monetary (RFM) behavior.

Business risks of this approach include:
- **Proxy Mismatch**: The proxy may not fully capture true default risk; some “disengaged” users might pay reliably if extended credit, while some active users might not.
- **Bias & Fairness**: Correlating disengagement with high risk could systematically exclude certain customer demographics unfairly.
- **Regulatory Scrutiny**: Regulators may question the validity of the proxy; if challenged, we must justify our choice with data and ensure non-discrimination.
- **Model Drift**: As the lending program matures and real default data becomes available, earlier models built on proxy targets may become obsolete, requiring retraining.

Proxies are a starting point, but must be updated with actual performance data as it is collected.

### What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?
- **Interpretable Models** (e.g., Logistic Regression with Weight of Evidence encoding):
  - **Pros**: Highly transparent, easy to explain, easier to validate and audit, regulatory friendly
  - **Cons**: May underperform compared to complex models if relationships in the data are highly non-linear or dependent on subtle interactions

- **Complex Models** (e.g., Gradient Boosting, Random Forests):
  - **Pros**: Often deliver superior predictive performance, can capture complex feature interactions, may reduce error rates on test data
  - **Cons**: Often “black boxes,” harder to explain decision logic, challenging to validate, may be scrutinized or rejected by regulators; require careful monitoring for drift or bias

**Key trade-off**: In regulated financial services, slight improvements in predictive power must be weighed against increased compliance, operational, and reputational risk. A simpler, well-documented, and interpretable model is often preferable for deployment—especially at program launch or when using proxy targets. More complex models can be considered for challenger approaches or when interpretability techniques (e.g., SHAP, LIME) are available and robust governance processes are in place.

---

## Project Structure

```
credit-risk-model/
├── .github/workflows/ci.yml   # For CI/CD
├── data/                      # add this folder to .gitignore
│   ├── raw/                   # Raw data goes here 
│   └── processed/             # Processed data for training
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── train.py
│   ├── predict.py
│   └── api/
│       ├── main.py
│       └── pydantic_models.py
├── tests/
│   └── test_data_processing.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

---
# Task 5 — Model Training & Experiment Tracking

## Overview
This project trains supervised models to predict a proxy risk label using
engineered features and tracks experiments using MLflow.

## How to Run

### 1. Start MLflow UI
```bash
mlflow ui
