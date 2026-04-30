# 📉 Customer Churn Prediction
### Identifying at-risk customers using Machine Learning on Telco subscriber data

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-orange?style=flat&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat)
![Domain](https://img.shields.io/badge/Domain-Telecom%20%7C%20Consulting-blueviolet?style=flat)

---

## 📌 Business Problem

A telecom company is losing customers faster than it is acquiring them. Every churned customer represents lost monthly revenue that is expensive to recover — acquiring a new customer costs 5–7x more than retaining an existing one.

The goal of this project is to **predict which customers are likely to churn** and **identify the key drivers** behind that decision — so the business can intervene early with targeted retention strategies.

---

## 🎯 Business Impact

- ✅ Identified the **top drivers of churn** — high monthly charges, month-to-month contracts, and lack of tech support — enabling targeted retention campaigns
- ✅ Built a model that flags **high-risk customers before they leave**, giving the business a window to intervene
- ✅ A 5% reduction in churn rate on this customer base could retain hundreds of customers and recover significant monthly recurring revenue
- ✅ Customers on **month-to-month contracts** were found to churn at 3x the rate of those on annual contracts — a direct actionable insight for the sales team
- ✅ Demonstrated that **high monthly charges + no online security + fiber optic internet** is the highest-risk customer profile



---

## 📂 Project Structure

```
customer-churn-prediction/
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Dataset (from Kaggle)
├── churn_analysis.ipynb                    # Full analysis notebook
└── README.md
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| Rows | 7,043 customers |
| Features | 21 (demographics, services, account info) |
| Target | Churn (Yes / No) |
| Churn Rate | ~26.5% |

---

## 🔍 Exploratory Data Analysis — Key Findings

**1. Contract Type is the Strongest Churn Predictor**
Customers on month-to-month contracts churn at ~42% — nearly 3x the rate of those on one or two-year contracts (~11% and ~3% respectively). Locking customers into longer contracts is the single most effective retention lever.

**2. High Monthly Charges Drive Churn**
Churned customers pay significantly higher monthly charges on average. Customers paying above $65/month show a steep increase in churn probability — suggesting a pricing sensitivity threshold.

**3. Tenure is Inversely Related to Churn**
New customers (0–12 months tenure) are at the highest risk. Customers who stay beyond 24 months rarely churn. Early engagement programs targeting new customers would have the highest ROI.

**4. Lack of Support Services Increases Churn**
Customers without tech support, online security, or online backup churn at roughly 2x the rate of those who have these services — indicating that value-added services improve stickiness.

---

## 🤖 Model Building & Results

| Model | Accuracy | F1 Score (Churn) | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 80% | 0.60 | 0.85 |
| Decision Tree | 78% | 0.57 | 0.73 |
| **Random Forest** ✅ | **82%** | **0.63** | **0.87** |
| Gradient Boosting | 81% | 0.62 | 0.86 |

**Random Forest** was selected as the final model based on best F1 Score and ROC-AUC.

---

## 🔑 Top Churn Drivers (Feature Importance)

| Rank | Feature | Business Meaning |
|---|---|---|
| 1 | Monthly Charges | Higher bills → more likely to leave |
| 2 | Tenure | Newer customers → higher risk |
| 3 | Contract Type | Month-to-month → 3x churn rate |
| 4 | Total Charges | Correlated with tenure |
| 5 | Tech Support | No support → 2x churn rate |

---

## 💡 Business Recommendations

Based on the analysis, three targeted actions would have the highest impact on reducing churn:

**1. Target High Monthly Charge Customers**
Customers paying above $65/month with less than 12 months tenure are the highest-risk segment. A personalised discount or loyalty reward at month 6 could significantly reduce early churn.

**2. Incentivise Contract Upgrades**
Since month-to-month customers churn at 42%, offering a discount to switch to an annual plan during the first 3 months could anchor customers long-term.

**3. Bundle Support Services for New Customers**
Free trials of tech support and online security for the first 6 months would increase perceived value and improve retention — especially for fiber optic internet customers.

---

## 📈 Evaluation Metrics Used

| Metric | Why It Was Used |
|---|---|
| F1 Score | Balances catching churners vs false alarms |
| ROC-AUC | Measures overall model discrimination ability |
| Confusion Matrix | Shows actual churn caught vs missed |
| Feature Importance | Reveals what is driving churn |

---

## 🚀 Model Deployment

The trained model is deployed as a REST API using Flask.

**Endpoint:** `POST /predict`

**Input (JSON):**
{
  "tenure": 12,
  "MonthlyCharges": 65.5,
  "Contract": "Month-to-month",
  ...
}

**Output (JSON):**
{
  "churn_prediction": 1,
  "churn_probability": 0.82
}

Run locally:
pip install flask
python app.py

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10 | Core language |
| Pandas & NumPy | Data manipulation |
| Matplotlib | Visualisation |
| Scikit-Learn | Model building & evaluation |
| Jupyter Notebook | Development environment |
| Flask | Model deployment as REST API |

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction

# 2. Install dependencies
pip install pandas numpy matplotlib scikit-learn jupyter

# 3. Launch the notebook
jupyter notebook churn_analysis.ipynb
```

---

## 💡 Key Learnings

- Contract type and monthly charges are far stronger churn predictors than demographic features like gender or age
- Churn is heavily concentrated in the **first 12 months** — early intervention has the highest retention ROI
- Feature importance from tree models directly translates to **actionable business recommendations** — this is what consulting firms like ZS and Deloitte care about most

---

## 👤 Author

Shreya Bhattacharjee
B.Tech — KIIT University
[LinkedIn](https://linkedin.com/in/shreya-bhattacharjee-b648a9228) • [GitHub](https://github.com/your-shrfr)
