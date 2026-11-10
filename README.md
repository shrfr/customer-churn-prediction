# 📉 Customer Churn Prediction
### Identifying at-risk customers using Machine Learning on Telco subscriber data

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9-orange?style=flat&logo=scikit-learn)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?style=flat&logo=fastapi)
![Domain](https://img.shields.io/badge/Domain-Telecom%20%7C%20Consulting-blueviolet?style=flat)

---

## 📌 Business Problem

A telecom company is losing customers faster than it is acquiring them. Every churned customer represents lost monthly revenue that is expensive to recover — acquiring a new customer typically costs 5–7x more than retaining an existing one.

The goal of this project is to **predict which customers are likely to churn** and **identify the key drivers** behind that decision, so the business can intervene early with targeted retention strategies.

---

## 🎯 Business Impact

- ✅ Identified the main drivers of churn — **month-to-month contracts, short tenure, no tech support/online security, and higher monthly charges** — enabling targeted retention campaigns
- ✅ Built a model that flags **high-risk customers before they leave** (it catches ~79% of customers who actually churn)
- ✅ Customers on **month-to-month contracts churn at ~43%**, versus ~11% on one-year and ~3% on two-year contracts — a direct, actionable insight for the sales team
- ✅ Customers in their **first 12 months churn at ~48%**, falling to ~7% after five years, so early engagement has the highest retention payoff
- ✅ On this customer base, preventing just 5% of churn would retain roughly 90 customers (5% of ~1,870 churners)

---

## 📂 Project Structure

```
customer-churn-prediction/
│
├── app/
│   └── main.py                 # FastAPI prediction service
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Dataset (from Kaggle)
├── model/
│   └── churn_model.joblib      # Trained pipeline (created by train.py)
├── notebooks/
│   ├── Churn Analysis-EDA.ipynb
│   └── Churn Prediction Model.ipynb
├── sample_inputs/
│   └── first_telc.csv          # Sample raw customer records
├── scripts/
│   └── train.py                # Training script
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| Rows | 7,043 customers (7,032 after removing 11 rows with blank TotalCharges) |
| Features | 19 model inputs (demographics, services, account info) |
| Target | Churn (Yes / No) |
| Churn Rate | ~26.6% |

---

## 🔍 Exploratory Data Analysis — Key Findings

**1. Contract Type is a Strong Churn Predictor**
Month-to-month customers churn at ~43%, versus ~11% (one-year) and ~3% (two-year). Moving customers onto longer contracts is one of the most effective retention levers.

**2. Higher Monthly Charges Go With Higher Churn**
Churned customers pay about $74 per month on average versus about $61 for retained customers. Customers paying more than $65/month churn at ~35%, roughly double the rate of those paying less (~16%).

**3. Tenure is Inversely Related to Churn**
Churn is ~48% in the first 12 months, ~29% in months 13–24, and drops steadily to ~7% for customers with 5+ years of tenure.

**4. Lack of Support Services Increases Churn**
Customers without tech support, online security or online backup churn at ~40–42%, versus ~15–22% for those who have them — close to 3x for tech support and online security.

**5. Fiber Optic Customers Churn More**
Fiber optic customers churn at ~42% versus ~19% for DSL, which suggests a price or service-quality issue worth investigating.

---

## 🤖 Modelling Approach

- **Preprocessing:** One-hot encoding of the 15 categorical features, with numeric features passed through, all inside one scikit-learn `Pipeline`, so the API accepts raw customer fields
- **Class imbalance:** Churners are only ~27% of the data, so **SMOTEENN** resampling is applied to the **training set only** (inside the pipeline, so it never touches test data or live predictions)
- **Model:** Random Forest (100 trees, `max_depth=6`, `min_samples_leaf=8`), chosen over a Decision Tree baseline
- **Evaluation:** Stratified 80/20 train/test split (1,407 test customers)

### Results (held-out test set)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Stayed (0) | 0.91 | 0.73 | 0.81 |
| **Churned (1)** | **0.52** | **0.79** | **0.62** |

Overall accuracy: **75%**

Confusion matrix:

| | Predicted stay | Predicted churn |
|---|---|---|
| **Actually stayed** | 755 | 278 |
| **Actually churned** | 78 | 296 |

**Trade-off:** The model is tuned toward **catching churners** (79% recall). The cost is false alarms: about half of the customers it flags would not have left. For retention campaigns, where a wasted discount is cheaper than a lost customer, that is usually acceptable.

---

## 🔑 Top Churn Drivers (Random Forest Feature Importance)

Importance summed across each feature's encoded categories:

| Rank | Feature | Business Meaning |
|---|---|---|
| 1 | Tech Support | No support → ~2.7x churn rate |
| 2 | Contract Type | Month-to-month → highest churn |
| 3 | Online Security | No security add-on → higher churn |
| 4 | Tenure | Newer customers → higher risk |
| 5 | Online Backup | No backup add-on → higher churn |

---

## 💡 Business Recommendations

**1. Target New, Higher-Paying Customers**
Customers paying more than $65/month with under 12 months of tenure churn at ~67% in this data. A personalised discount or loyalty offer around month 6 could reduce early churn.

**2. Incentivise Contract Upgrades**
Month-to-month customers churn at ~43%. Offering a discount to move to an annual plan in the first few months could anchor customers long-term.

**3. Bundle Support Services for New Customers**
Free trials of tech support and online security for the first six months would increase perceived value, especially for fiber optic customers.

---

## 🚀 Model Deployment

The trained pipeline is served as a REST API with **FastAPI**.

**Endpoint:** `POST /predict` (interactive docs at `/docs`)

**Example request:**
```json
{
  "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
  "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
  "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
  "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
  "StreamingMovies": "No", "Contract": "Month-to-month",
  "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
  "MonthlyCharges": 29.85, "TotalCharges": 29.85
}
```

**Example response:**
```json
{
  "churn_probability": 0.763,
  "churn_prediction": 1,
  "risk_level": "high"
}
```

Risk levels: `low` (< 0.3), `medium` (0.3–0.6), `high` (≥ 0.6). These cut-offs are illustrative and can be tuned to the business's retention capacity. Invalid inputs (for example an unknown contract type) are rejected with a validation error.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.12 | Core language |
| Pandas & NumPy | Data manipulation |
| Matplotlib & Seaborn | Visualisation |
| Scikit-Learn | Pipeline, encoding, Random Forest |
| imbalanced-learn | SMOTEENN resampling |
| FastAPI & Uvicorn | REST API |
| Jupyter Notebook | Analysis and experimentation |

---

## ▶️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/shrfr/customer-churn-prediction.git
cd customer-churn-prediction

# 2. Create an environment and install dependencies
python -m venv .venv
.venv\Scripts\activate          # Windows (use source .venv/bin/activate on Mac/Linux)
pip install -r requirements.txt

# 3. Train the model (creates model/churn_model.joblib)
python scripts/train.py

# 4. Start the API
uvicorn app.main:app --reload
# Open http://127.0.0.1:8000/docs
```



To explore the analysis, open the notebooks in the `notebooks/` folder with Jupyter.

---

## ⚠️ Limitations

- Trained on a single public dataset; real-world performance would need monitoring and retraining
- Precision on churners is ~52%, so roughly half of flagged customers are false alarms
- The model predicts risk, not the cause; business recommendations come from the EDA and feature importance, not from causal analysis

---

## 💡 Key Learnings

- Contract type, tenure and add-on services (tech support, online security) are much stronger churn signals than demographics like gender
- Churn is heavily concentrated in the **first 12 months**, so early intervention has the highest retention payoff
- Putting preprocessing and resampling **inside one pipeline** prevents train/serve mismatch and data leakage, and makes the model deployable as an API
- Feature importance from tree models translates directly into actionable business recommendations

---

## 👤 Author

Shreya Bhattacharjee
B.Tech — KIIT University
[LinkedIn](https://linkedin.com/in/shreya-bhattacharjee-b648a9228) • [GitHub](https://github.com/shrfr)
