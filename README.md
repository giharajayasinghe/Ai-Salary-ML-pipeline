# AI Salary Prediction (2026 Dataset)

This project predicts **AI job salaries in USD** using machine learning models based on job role, experience level, company details, and work year.

The dataset contains over **150,000 AI job records** and reflects modern AI job market trends up to 2026.

---

## 📊 Dataset
Source:  
https://github.com/foorilla/ai-jobs-net-salaries

Target variable:
- `salary_in_usd`

Key features:
- Work year
- Experience level
- Employment type
- Job title
- Remote ratio
- Company size and location

---

## 🧠 Models Used
- Linear Regression  
- Ridge Regression  
- Lasso Regression  
- Random Forest Regressor  

All models are trained using **Scikit-learn Pipelines** with proper preprocessing.

---

## ⚙️ Preprocessing
- OneHotEncoding for nominal categorical features
- Standard Scaling for ordinal features (`work_year`)
- Train-test split (80/20)
- Evaluation using:
  - MAE
  - RMSE
  - R² score

---

## 📈 Results (Test Set)
| Model | MAE | RMSE | R² |
|------|-----|------|----|
| Linear Regression | ~45k | ~62k | ~0.29 |
| Ridge Regression | ~45k | ~62k | ~0.29 |
| Lasso Regression | ~45k | ~62k | ~0.29 |
| Random Forest | ~44k | ~61k | ~0.31 |

---

## 🛠 Technologies
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

---

## ▶️ How to Run
```bash
pip install -r requirements.txt
jupyter notebook
