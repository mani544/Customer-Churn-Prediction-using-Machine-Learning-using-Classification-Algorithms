import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import os

# ------------------------------------------------------
# 1. Page Configuration (Must be first)
# ------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction Lab",
    page_icon="📊",
    layout="wide"
)

# ------------------------------------------------------
# 2. Imports & Dependency Handling
# ------------------------------------------------------
# Try importing optional visual libraries
try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from streamlit_lottie import st_lottie
    LOTTIE_AVAILABLE = True
except ImportError:
    LOTTIE_AVAILABLE = False

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Machine Learning Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# xgboost might not be installed in some environments; keep import as-is
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False
    # Provide a lightweight fallback classifier if needed
    XGBClassifier = None

# ------------------------------------------------------
# 3. CSS & UI Styling (Merged CSS from both sources)
# ------------------------------------------------------
def local_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

        /***** Combined Inline CSS (previous local_css + assets/style.css) *****/

        /* Fix header cutoff */
        .block-container {
            padding-top: 2rem !important;
        }

        /* App Background (merged) */
        .stApp {
            /* gradient background from assets/style.css for richer look */
            background: linear-gradient(180deg, #071133 0%, #08132a 100%);
            font-family: 'Inter', sans-serif;
            color: #e6eef8;
        }

        /* Container padding (from assets/style.css) */
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }

        /* Custom Card Style (kept dark card style) */
        .metric-card {
            background-color: #262730;
            border: 1px solid #464b5d;
            border-radius: 8px;
            padding: 15px;
            color: white;
        }

        .card {
            background: rgba(255,255,255,0.03);
            border-radius: 14px;
            padding: 16px;
            box-shadow: 0 8px 30px rgba(3,7,18,0.6);
        }

        .big-title {
            font-size: 2.2rem;
            font-weight: 800;
            color: white;
            margin-bottom: 0px;
        }

        .muted {
            color: #9fb0cf;
            font-size: 1rem;
        }

        /* Dataframe / table fallback */
        .css-1b3pqpe { /* streamlit table container class may vary */
            background: rgba(255,255,255,0.02);
        }

        /* Metric card colors (ensure metrics are visible) */
        .stMetricLabel, .stMetricValue {
            color: #e6eef8 !important;
        }

        /* Button Styling (merged gradients) */
        div.stButton > button {
            background: linear-gradient(90deg, #00b8d9, #ff8c42);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 8px 12px;
            font-weight: 600;
        }
        div.stButton > button:hover {
            border: 1px solid white;
        }

        /* Smaller buttons (original) */
        div.stButton > button {
            padding: 0.5rem 1rem;
            border-radius: 6px;
        }

        /* Footer */
        .footer {
            color: #b6c2d6;
            font-size: 12px;
            opacity: 0.9;
        }

        /* Misc adjustments */
        .streamlit-expanderHeader {
            color: #dbe9ff;
        }

        /* Keep charts readable */
        .plotly-graph-div .main-svg {
            background: transparent;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------
# 4. Helper Functions
# ------------------------------------------------------
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=3)
        if r.status_code == 200:
            return r.json()
    except:
        return None
    return None

def render_header():
    col1, col2 = st.columns([1, 5])

    # Logo (safely handle if missing)
    with col1:
        if Image and os.path.exists("assets/logo.png"):
            st.image("assets/logo.png", width=80)
        else:
            # Placeholder emoji if no logo image found
            st.markdown("# 📊")

    # Titles
    with col2:
        st.markdown("<div class='big-title'>Customer Churn Prediction ML Lab</div>", unsafe_allow_html=True)
        st.markdown("<div class='muted'>Model Comparison • XGBoost • SVM • Real-time Inference</div>",
                    unsafe_allow_html=True)

    # Lottie Animation (if available)
    if LOTTIE_AVAILABLE:
        lottie_url = "https://assets3.lottiefiles.com/packages/lf20_jcikwtux.json"
        lottie_json = load_lottieurl(lottie_url)
        if lottie_json:
            try:
                st_lottie(lottie_json, height=150, key="header_lottie")
            except Exception:
                pass

# ------------------------------------------------------
# 5. Core Machine Learning Logic
# ------------------------------------------------------
def train_churn_models(df: pd.DataFrame):
    df = df.copy()

    # 1. Cleanup
    if "CustomerID" in df.columns:
        df = df.drop(columns=["CustomerID"])

    # 2. Encode Target
    if df["Churn"].dtype == "O":
        df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    # 3. Encode Categorical Features
    cat_cols = ["Gender", "Subscription Type", "Contract Length"]
    # Only encode columns that actually exist in the dataframe
    cat_cols = [c for c in cat_cols if c in df.columns]

    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # 4. Clean Bool columns from get_dummies
    bool_cols = df_encoded.select_dtypes(include=["bool"]).columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype("int64")

    # 5. Define X and y
    X = df_encoded.drop("Churn", axis=1)
    y = df_encoded["Churn"]
    feature_cols = X.columns.tolist()

    # 6. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 7. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 8. Define Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(probability=True, random_state=42),
    }

    if XGBOOST_AVAILABLE and XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
    else:
        # If xgboost not available, provide warning metric later but continue
        st.warning("XGBoost not available in this environment — skipping XGBoost model.")

    # Models that require scaling
    scaled_models = ["Logistic Regression", "KNN", "Naive Bayes", "SVM"]

    results = []
    trained_models = {}

    # 9. Train Loop
    progress_bar = st.progress(0)
    total_models = len(models)

    for idx, (name, model) in enumerate(models.items()):
        use_scaled = name in scaled_models
        X_tr = X_train_scaled if use_scaled else X_train
        X_te = X_test_scaled if use_scaled else X_test

        model.fit(X_tr, y_train)
        trained_models[name] = model

        y_pred = model.predict(X_te)

        # Get Probabilities (Handle edge cases)
        y_prob = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_te)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_te)
            # Protect against constant scores
            if scores.max() != scores.min():
                y_prob = (scores - scores.min()) / (scores.max() - scores.min())
            else:
                y_prob = np.zeros_like(scores)

        # ROC_AUC requires probability-like scores; if none available default to 0.5
        try:
            roc_val = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.5
        except Exception:
            roc_val = 0.5

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "ROC_AUC": roc_val,
        }
        results.append(metrics)
        progress_bar.progress((idx + 1) / total_models)

    progress_bar.empty()

    results_df = pd.DataFrame(results).sort_values(by="ROC_AUC", ascending=False).reset_index(drop=True)
    best_model_name = results_df.iloc[0]["Model"]

    # Pack test sets for visualization later
    test_sets = (X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)

    return trained_models, scaler, feature_cols, results_df, best_model_name, scaled_models, test_sets

def build_feature_row(feature_cols, age, tenure, usage_freq, support_calls, payment_delay, total_spend,
                      last_interaction, gender, subscription_type, contract_length):
    # Create a DataFrame with zeros
    row = pd.DataFrame(0, index=[0], columns=feature_cols)

    # Manual mapping based on inputs
    mapping = {
        "Age": age,
        "Tenure": tenure,
        "Usage Frequency": usage_freq,
        "Support Calls": support_calls,
        "Payment Delay": payment_delay,
        "Total Spend": total_spend,
        "Last Interaction": last_interaction,
    }

    # Fill numeric data
    for k, v in mapping.items():
        if k in row.columns:
            row.at[0, k] = v

    # One-Hot Encoding Logic (Manually matching drop_first=True)
    if gender == "Male" and "Gender_Male" in row.columns:
        row.at[0, "Gender_Male"] = 1

    if subscription_type == "Standard" and "Subscription Type_Standard" in row.columns:
        row.at[0, "Subscription Type_Standard"] = 1
    elif subscription_type == "Premium" and "Subscription Type_Premium" in row.columns:
        row.at[0, "Subscription Type_Premium"] = 1
    elif subscription_type == "Basic" and "Subscription Type_Basic" in row.columns:
        row.at[0, "Subscription Type_Basic"] = 1

    if contract_length == "Monthly" and "Contract Length_Monthly" in row.columns:
        row.at[0, "Contract Length_Monthly"] = 1
    elif contract_length == "Quarterly" and "Contract Length_Quarterly" in row.columns:
        row.at[0, "Contract Length_Quarterly"] = 1
    elif contract_length == "Annual" and "Contract Length_Annual" in row.columns:
        row.at[0, "Contract Length_Annual"] = 1

    return row

def plot_roc_bar(results_df):
    if PLOTLY_AVAILABLE:
        fig = px.bar(
            results_df,
            x="ROC_AUC",
            y="Model",
            orientation="h",
            color="ROC_AUC",
            color_continuous_scale="Viridis",
            title="Model ROC AUC Comparison",
            text="ROC_AUC",
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="inside")
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(results_df.set_index("Model")["ROC_AUC"])

def show_confusion_matrices(models, X_test, X_test_scaled, y_test, scaled_models):
    n = len(models)
    cols = 3
    rows = (n + cols - 1) // cols

    # Adjust figure size dynamically
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = axes.flatten()

    for idx, (name, model) in enumerate(models.items()):
        use_scaled = name in scaled_models
        X_te = X_test_scaled if use_scaled else X_test

        y_pred = model.predict(X_te)

        # Plot to the specific axis
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            ax=axes[idx],
            colorbar=False,
            cmap="Blues"
        )
        axes[idx].set_title(name)
        axes[idx].grid(False)

    # Remove empty subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    st.pyplot(fig)

# ------------------------------------------------------
# 6. Main Execution
# ------------------------------------------------------
def main():
    local_css()
    render_header()

    st.markdown("---")

    # --- SIDEBAR ---
    st.sidebar.header("1. Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload Churn CSV", type=["csv"])

    st.sidebar.markdown("---")
    st.sidebar.header("2. Control Panel")

    # Session State Initialization
    if "models" not in st.session_state:
        st.session_state.models = None

    # Data Loading & Training
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"Loaded: {len(df)} rows")

            with st.expander("📊 Preview Data", expanded=False):
                st.dataframe(df.head())

            if st.sidebar.button("Train All Models", type="primary"):
                with st.spinner("Training models & comparing metrics..."):
                    # Call training function
                    (
                        trained_models, scaler, feature_cols, results_df,
                        best_model_name, scaled_models, test_sets
                    ) = train_churn_models(df)

                    # Store in session state
                    st.session_state.models = trained_models
                    st.session_state.scaler = scaler
                    st.session_state.feature_cols = feature_cols
                    st.session_state.results_df = results_df
                    st.session_state.best_model_name = best_model_name
                    st.session_state.scaled_models = scaled_models
                    st.session_state.test_sets = test_sets

                st.success(f"Training Complete! Best Model: **{best_model_name}**")
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.info("👋 Please upload a CSV file to begin analysis.")
        # Create a sample dataframe button for demo purposes
        if st.sidebar.button("Use Demo Data"):
            # Create synthetic data if no file is uploaded
            np.random.seed(42)
            data_size = 500
            demo_data = pd.DataFrame({
                'Age': np.random.randint(18, 70, data_size),
                'Gender': np.random.choice(['Male', 'Female'], data_size),
                'Tenure': np.random.randint(1, 60, data_size),
                'Usage Frequency': np.random.randint(1, 30, data_size),
                'Support Calls': np.random.randint(0, 10, data_size),
                'Payment Delay': np.random.randint(0, 30, data_size),
                'Subscription Type': np.random.choice(['Basic', 'Standard', 'Premium'], data_size),
                'Contract Length': np.random.choice(['Monthly', 'Quarterly', 'Annual'], data_size),
                'Total Spend': np.random.randint(100, 5000, data_size),
                'Last Interaction': np.random.randint(1, 100, data_size),
                'Churn': np.random.choice(['Yes', 'No'], data_size, p=[0.2, 0.8])
            })

            # Trigger Training on Demo Data
            (
                trained_models, scaler, feature_cols, results_df,
                best_model_name, scaled_models, test_sets
            ) = train_churn_models(demo_data)

            st.session_state.models = trained_models
            st.session_state.scaler = scaler
            st.session_state.feature_cols = feature_cols
            st.session_state.results_df = results_df
            st.session_state.best_model_name = best_model_name
            st.session_state.scaled_models = scaled_models
            st.session_state.test_sets = test_sets
            st.success("Trained on Demo Data!")

    # --- RESULTS SECTION ---
    if st.session_state.models is not None:

        # Tabs for better organization
        tab1, tab2, tab3 = st.tabs(["📈 Model Comparison", "😵 Confusion Matrix", "🎯 Prediction Lab"])

        with tab1:
            st.markdown("### Performance Metrics")
            # Show the best model metric in a metric card
            best_row = st.session_state.results_df.iloc[0]
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Best Model", best_row['Model'])
            col_b.metric("Best ROC_AUC", f"{best_row['ROC_AUC']:.4f}")
            col_c.metric("Best Accuracy", f"{best_row['Accuracy']:.4f}")

            st.dataframe(st.session_state.results_df.style.highlight_max(axis=0, color='#1f77b4'),
                         use_container_width=True)
            plot_roc_bar(st.session_state.results_df)

        with tab2:
            st.markdown("### Confusion Matrices")
            X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = st.session_state.test_sets
            show_confusion_matrices(
                st.session_state.models, X_test, X_test_scaled, y_test, st.session_state.scaled_models
            )

        with tab3:
            st.markdown("### Simulator")
            st.write("Adjust parameters below to predict if a customer will churn.")

            # Model Selection
            model_names = list(st.session_state.models.keys())
            # Default to best model
            default_idx = model_names.index(st.session_state.best_model_name)
            selected_model = st.selectbox("Choose Model for Inference", model_names, index=default_idx)

            # Input Form
            with st.form("prediction_form"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    age = st.number_input("Age", 18, 100, 30)
                    gender = st.selectbox("Gender", ["Male", "Female"])
                    tenure = st.number_input("Tenure (months)", 0, 240, 12)

                with col2:
                    usage = st.number_input("Usage Frequency", 0, 100, 10)
                    support = st.number_input("Support Calls", 0, 50, 1)
                    delay = st.number_input("Payment Delay", 0, 60, 0)

                with col3:
                    spend = st.number_input("Total Spend ($)", 0, 100000, 500)
                    last = st.number_input("Days Since Last Interaction", 0, 365, 10)
                    sub = st.selectbox("Subscription", ["Basic", "Standard", "Premium"])
                    cont = st.selectbox("Contract", ["Annual", "Monthly", "Quarterly"])

                submit_btn = st.form_submit_button("Predict Churn Probability")

            if submit_btn:
                # Build Row
                row = build_feature_row(
                    st.session_state.feature_cols,
                    age, tenure, usage, support, delay, spend, last, gender, sub, cont
                )

                model = st.session_state.models[selected_model]

                # Scale if necessary
                if selected_model in st.session_state.scaled_models:
                    row_processed = st.session_state.scaler.transform(row)
                else:
                    row_processed = row

                # Predict
                if hasattr(model, "predict_proba"):
                    pred_prob = model.predict_proba(row_processed)[0][1]
                else:
                    # Fallback for models without proba
                    try:
                        d = model.decision_function(row_processed)
                        pred_prob = 1 / (1 + np.exp(-d))[0]  # Sigmoid approximation
                    except Exception:
                        pred_prob = 0.5

                pred_class = model.predict(row_processed)[0]

                # Display Result
                st.markdown("---")
                c1, c2 = st.columns([1, 2])

                with c1:
                    st.metric("Churn Probability", f"{pred_prob:.1%}")

                with c2:
                    if pred_class == 1:
                        st.error(f"⚠️ **High Risk**: This customer is likely to churn.")
                    else:
                        st.success(f"✅ **Low Risk**: This customer is likely to stay.")

if __name__ == "__main__":
    main()
