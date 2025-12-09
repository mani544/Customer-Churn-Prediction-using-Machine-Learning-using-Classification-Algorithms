

---

# 🚀 **Customer Churn Prediction – Machine Learning Project**

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Modeling-orange?logo=scikitlearn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-lightgrey?logo=numpy)
![Streamlit](https://img.shields.io/badge/Streamlit-App%20UI-FF4B4B?logo=streamlit)
![Status](https://img.shields.io/badge/Project-Active-success)
![License](https://img.shields.io/badge/License-MIT-green)

</p>

---

# 📌 **Project Overview**

Customer churn is one of the most critical business challenges for telecom companies.
This machine learning project predicts **whether a customer will churn** based on their behavior, contract details, usage patterns, and service data.

This repository includes:

✔ End-to-end data cleaning
✔ Exploratory data analysis
✔ Feature engineering
✔ Model training & evaluation
✔ Hyperparameter tuning using **GridSearchCV**
✔ Saving the best model with Joblib
✔ Streamlit UI for deployment

---

# 🧠 **Problem Statement**

Predict whether a customer will **leave the service** (churn) using classification algorithms.

This model helps companies:

* Identify high-risk customers
* Take preventive retention actions
* Improve customer satisfaction
* Reduce revenue loss

---

# 📂 **Project Structure**

```
Customer-Churn-Prediction/
│
├── data/
│   ├── churn_data.csv
│
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│
├── notebooks/
│   ├── customer_churn.ipynb
│
├── streamlit_app/
│   ├── app_streamlit.py
│   ├── assets/
│       ├── logo.png
│       ├── animations.json
│
├── utils/
│   ├── preprocess.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 📊 **Exploratory Data Analysis (EDA)**

EDA included:

* Missing value treatment
* Outlier detection
* Churn distribution
* Demographic analysis
* Contract type vs churn
* Services used vs churn

Key visualizations:

✔ Count plots
✔ Correlation heatmap
✔ Feature importance
✔ Tenure vs churn

---

# 🧬 **Feature Engineering**

Major steps:

* Label Encoding
* One-Hot Encoding
* Scaling numeric values
* Dropping irrelevant features
* Converting categorical variables

---

# 🤖 **Models Used**

We experimented with the following algorithms:

| Model                    | Used           |
| ------------------------ | -------------- |
| Logistic Regression      | ✔              |
| Random Forest Classifier | ✔ (BEST MODEL) |
| KNN                      | ✔              |
| Naive Bayes              | ✔              |
| XGBoost                  | (Optional)     |

---

# 🔍 **Hyperparameter Tuning (GridSearchCV)**

The best model selected = **Random Forest Classifier**

### ✔ GridSearchCV Code

```python
from sklearn.model_selection import GridSearchCV

params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [4, 6, 8, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)

grid = GridSearchCV(
    rf,
    params,
    cv=3,
    scoring="roc_auc",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print(grid.best_params_)
print("Best ROC AUC:", grid.best_score_)
```

### ✔ Best Parameters (example)

```json
{
  "n_estimators": 200,
  "max_depth": 8,
  "min_samples_split": 5,
  "min_samples_leaf": 2
}
```

---

# 💾 **Saving the Best Model**

```python
import joblib
joblib.dump(model, "models/best_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
```

---

# 🌐 **Streamlit App**

This project includes an interactive UI built using Streamlit:

Features:

* Customer details form
* Prediction output
* Clean animations (Lottie)
* Styling with custom CSS
* Model probability display

---

# 🛠 **Installation & Setup**

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Jupyter Notebook

```bash
jupyter notebook
```

### 4️⃣ Run Streamlit App

```bash
streamlit run streamlit_app/app_streamlit.py
```

---

# 🎯 **Model Evaluation Metrics**

| Metric    | Score                             |
| --------- | --------------------------------- |
| Accuracy  | ~80–85%                           |
| ROC-AUC   | ~0.88                             |
| Precision | High for churn class              |
| Recall    | Prioritized for business use-case |

---

# 🏁 **Final Output**

✔ Displays churn probability
✔ Predicts churn with optimized parameters
✔ Business-friendly UI
✔ Ready for deployment

---

# 🧪 **Technologies Used**

* Python
* Pandas, NumPy
* Scikit-Learn
* Matplotlib / Seaborn
* Streamlit
* Joblib
* Lottie animations

---

# 🎓 **What I Learned**

* End-to-end ML workflow
* Handling imbalance
* Hyperparameter tuning (GridSearchCV, Random Search CV)
* Feature engineering best practices
* Deploying ML models with Streamlit

---

# 🤝 **Contributions**

Pull requests are welcome!

---

# 📜 License

This project is licensed under the **MIT License**.

---


