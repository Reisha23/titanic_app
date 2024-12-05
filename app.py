import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('titanic_model.pkl')

# Sidebar
st.sidebar.title("Prediksi kelangsungan hidup Passenger Titanic")
st.sidebar.markdown("Dataset: [Kaggle Titanic](https://www.kaggle.com/competitions/titanic/data)")
st.sidebar.markdown("Author: Reisha Narindra Whibangga")

# Header
st.title("Titanic Prediksi Kelangsungan Hidup")

# View Dataset
st.subheader("Dataset Overview")
data = pd.read_csv('data/train.csv')
st.write(data.head())

# Visualization
st.subheader("Visualisasi")
st.bar_chart(data['Survived'].value_counts())

# Prediction Form
st.subheader("Prediksi")
Pclass = st.selectbox("Passenger Class", [1, 2, 3])
Sex = st.selectbox("Sex", ["Male", "Female"])
Age = st.slider("Age", 0, 80, 25)
SibSp = st.slider("Siblings/Spouses Aboard", 0, 8, 0)
Parch = st.slider("Parents/Children Aboard", 0, 6, 0)
Fare = st.number_input("Fare", 0.0, 500.0, 30.0)
Embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

if st.button("Predict"):
    # Convert inputs
    sex = 0 if Sex == "Male" else 1
    embarked = ["C", "Q", "S"].index(Embarked)
    features = [[Pclass, sex, Age, SibSp, Parch, Fare, embarked]]
    
    # Make prediction
    prediction = model.predict(features)
    result = "Survived" if prediction[0] == 1 else "Not Survived"
    st.success(f"Prediction: {result}")
