import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_scaled, y)

# --- Attractive Template Design ---
st.set_page_config(page_title="Iris Species Predictor", page_icon="🌸", layout="centered", initial_sidebar_state="auto")

# Header with background color and style
st.markdown("""
    <style>
    .main-header {
        background-color: #6C63FF;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-family: Arial, sans-serif;
    }
    </style>
    <div class="main-header">
        <h1>Iris Species Predictor 🌸</h1>
    </div>
    """, unsafe_allow_html=True)

st.write("---")

# Sidebar with instructions
st.sidebar.header("Features Input")
st.sidebar.write("Please enter the features below to predict the Iris species:")

# Input features from user in the sidebar
sepal_length = st.sidebar.slider('Sepal Length (cm)', min_value=4.0, max_value=8.0, step=0.1)
sepal_width = st.sidebar.slider('Sepal Width (cm)', min_value=2.0, max_value=5.0, step=0.1)
petal_length = st.sidebar.slider('Petal Length (cm)', min_value=1.0, max_value=7.0, step=0.1)
petal_width = st.sidebar.slider('Petal Width (cm)', min_value=0.1, max_value=2.5, step=0.1)

# Prediction button
if st.sidebar.button('Predict 🌟'):
    new_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled)
    
    st.success(f'Predicted Species: **{iris.target_names[prediction][0]}**')
    st.balloons()

# Footer with additional information
st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #6C63FF;
        color: white;
        text-align: center;
        padding: 10px;
        font-family: Arial, sans-serif;
    }
    </style>
    <div class="footer">
        <p>Developed by Orinson Technologies | Iris Species Predictor</p>
    </div>
    """, unsafe_allow_html=True)
