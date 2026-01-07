import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Telco Customer Churn",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- SIDEBAR ----------
st.sidebar.title("Telco Churn App")
st.sidebar.markdown("Logistic Regression Model")
st.sidebar.divider()

show_data = st.sidebar.checkbox("Show Dataset Preview", value=True)

# ---------- TITLE ----------
st.title("📊 Logistic Regression")
st.caption("Predict Customer Churn using Telco Customer Data")

# ---------- LOAD DATA ----------
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

# ---------- DATA PREVIEW ----------
if show_data:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

# ---------- TARGET ----------
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# ---------- CHURN DISTRIBUTION ----------
st.subheader("Churn Distribution")

col1, col2 = st.columns([2, 3])

with col1:
    churn_counts = df["Churn"].value_counts()
    fig, ax = plt.subplots()
    churn_counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Churn (0 = Stay, 1 = Leave)")
    ax.set_ylabel("Count")
    st.pyplot(fig)

with col2:
    st.markdown(
        """
        **Insight**
        - Majority of customers do **not churn**
        - Dataset is **imbalanced**
        - Logistic Regression is suitable for this task
        """
    )

# ---------- FEATURE ENGINEERING ----------
X = df.drop("Churn", axis=1)
y = df["Churn"]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------- MODEL ----------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ---------- CONFUSION MATRIX ----------
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
im = ax.imshow(cm)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12)

st.pyplot(fig)

# ---------- METRICS ----------
st.subheader("Model Performance")

c1, c2, c3 = st.columns(3)

c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred)*100:.2f}%")
c2.metric("Correct Churn Identified", cm[1][1])
c3.metric("Non-Churn Misclassified", cm[0][1])

# ---------- PREDICTION ----------
st.subheader("🔮 Predict Customer Churn")

p1, p2, p3 = st.columns(3)

with p1:
    tenure = st.slider("Tenure (Months)", 0, 72, 12)

with p2:
    monthly_charges = st.slider("Monthly Charges", 18.0, 120.0, 70.0)

with p3:
    total_charges = tenure * monthly_charges
    st.metric("Total Charges", f"{total_charges:.2f}")

sample = X.iloc[[0]].copy()
sample[:] = 0
sample["tenure"] = tenure
sample["MonthlyCharges"] = monthly_charges
sample["TotalCharges"] = total_charges

sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)[0]

st.divider()

if prediction == 1:
    st.error("⚠️ Customer is likely to **CHURN**")
else:
    st.success("✅ Customer is likely to **STAY**")
