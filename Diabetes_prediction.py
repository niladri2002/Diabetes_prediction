import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

# Function to initialize parameters
def initialize_parameters(n):
    w = np.zeros(n)
    b = 0.0
    return w, b

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic regression model
def logistic_regression(X, y, alpha, num_epochs):
    m, n = X.shape
    w, b = initialize_parameters(n)
    costs = []
    for epoch in range(num_epochs):
        dw = np.zeros(n)
        db = 0
        cost = 0
        for i in range(m):
            z = np.dot(w, X[i]) + b
            a = sigmoid(z)
            dz = a - y[i]
            dw += X[i] * dz
            db += dz
            cost += - (y[i] * np.log(a) + (1 - y[i]) * np.log(1 - a))
        cost /= m
        dw /= m
        db /= m
        w -= alpha * dw
        b -= alpha * db
        costs.append(cost)
        if epoch % 100 == 0:
            st.write(f'Epoch {epoch + 1}, Cost: {cost}, w1: {w[0]}, w2: {w[1]}')
    return w, b, costs

# Function to predict outcomes
def predict(X, w, b):
    z = np.dot(X, w) + b
    a = sigmoid(z)
    return [1 if i > 0.5 else 0 for i in a]

# Load data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    df = pd.read_csv(url)
    return df

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
    }
    .main {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stDataFrame {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", ["Home", "Train Model", "Make Prediction"])

# Load dataset
df = load_data()

if options == "Home":
    st.title("Pima Indian Diabetes Prediction")
    st.write("""
    ### Objective
    Predict whether a patient has diabetes based on diagnostic measures.
    """)
    st.write("### Dataset")
    st.write(df.head())

elif options == "Train Model":
    st.title("Train Model")
    st.write("### Dataset")
    st.write(df.head())

    # Data preprocessing
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    X = df[features]
    y = df['Outcome'].values  # Convert to NumPy array to avoid KeyError

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    st.write("### Feature Matrix (X)")
    st.write(X_scaled)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train the model
    alpha = 0.01
    num_epochs = 2000
    w, b, costs = logistic_regression(X_train, y_train, alpha, num_epochs)

    # Store trained parameters in session state
    st.session_state['scaler'] = scaler
    st.session_state['w'] = w
    st.session_state['b'] = b

    # Plot the cost function
    st.write("### Loss over epochs")
    fig, ax = plt.subplots()
    ax.plot(range(1, num_epochs + 1), costs)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss over epochs')
    st.pyplot(fig)

    # Evaluate the model
    y_pred = predict(X_test, w, b)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    st.write("### Model Performance")
    st.write(f"Accuracy: {accuracy}")
    st.write(f"Precision: {precision}")
    st.write(f"Recall: {recall}")
    st.write(f"F1 Score: {f1}")
    st.write(f"ROC AUC Score: {roc_auc}")

elif options == "Make Prediction":
    st.title("Predict Diabetes")
    st.write("Enter the patient details below:")

    # User input for prediction
    pregnancies = st.number_input("Pregnancies", min_value=0)
    glucose = st.number_input("Glucose", min_value=0)
    blood_pressure = st.number_input("BloodPressure", min_value=0)
    skin_thickness = st.number_input("SkinThickness", min_value=0)
    insulin = st.number_input("Insulin", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0, format="%.2f")
    dpf = st.number_input("DiabetesPedigreeFunction", min_value=0.0, format="%.2f")
    age = st.number_input("Age", min_value=0)

    # Predict button
    if st.button("Predict"):
        user_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        scaler = st.session_state.get('scaler')
        w = st.session_state.get('w')
        b = st.session_state.get('b')
        if scaler is not None and w is not None and b is not None:
            user_input_scaled = scaler.transform(user_input)
            prediction = predict(user_input_scaled, w, b)
            if prediction[0] == 1:
                st.write("The model predicts that the patient has diabetes.")
            else:
                st.write("The model predicts that the patient does not have diabetes.")
        else:
            st.write("Please train the model first.")
