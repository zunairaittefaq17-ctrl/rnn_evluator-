import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Student Predictor", layout="centered")

FEATURES = ['attendance','assignment','quiz','study_hours']

# ---------- AUTO TRAIN ----------
def train_model():
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    data = []
    for i in range(100):
        weeks = np.random.randint(50, 90, (5,4))
        result = 1 if weeks.mean() > 65 else 0
        data.append((weeks.flatten(), result))

    X = np.array([d[0] for d in data])
    y = np.array([d[1] for d in data])

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=300)
    model.fit(X, y)

    joblib.dump(model, "model.joblib")
    joblib.dump(scaler, "scaler.joblib")

    return model, scaler

# ---------- LOAD ----------
if not os.path.exists("model.joblib"):
    model, scaler = train_model()
else:
    model = joblib.load("model.joblib")
    scaler = joblib.load("scaler.joblib")

# ---------- UI ----------
st.title("🧠 Student Performance Predictor (RNN Style)")

weekly = []

for i in range(5):
    st.subheader(f"Week {i+1}")
    col1, col2 = st.columns(2)

    with col1:
        att = st.slider(f"Attendance {i+1}", 0,100,70)
        ass = st.slider(f"Assignment {i+1}", 0,100,70)

    with col2:
        quiz = st.slider(f"Quiz {i+1}", 0,100,70)
        hrs = st.slider(f"Study Hours {i+1}", 0,15,5)

    weekly.append([att, ass, quiz, hrs])

if st.button("Predict"):
    X = np.array(weekly).flatten().reshape(1,-1)
    X = scaler.transform(X)

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]

    if pred == 1:
        st.success(f"✅ PASS ({round(prob[1]*100,2)}%)")
    else:
        st.error(f"❌ FAIL ({round(prob[0]*100,2)}%)")
