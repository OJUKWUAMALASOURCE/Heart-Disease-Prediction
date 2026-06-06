# ❤️ Heart Disease Prediction App

This is a machine learning project that predicts whether a person is likely to have heart disease based on medical input features.

---

## 📌 Project Description

The model is trained using a heart disease dataset and uses **Random Forest** with feature scaling to make predictions.

A simple **Streamlit web app** is used for user interaction.

---

## 📂 Project Files

- `train_model.py` → Trains the machine learning model and saves it
- `app.py` → Streamlit web application for prediction
- `eda.py` → Exploratory data analysis of dataset
- `model.pkl` → Saved trained model
- `scaler.pkl` → Saved StandardScaler for preprocessing
- `heart.csv` → Dataset used for training
- `requirements.txt` → Required Python libraries
- `README.md` → Project documentation

---

## ⚙️ How to Run This Project

### 1. Install dependencies# ❤️
---

### 2. Train the model
This will generate:
- `model.pkl`
- `scaler.pkl`

---

### 3. Run the web app
---

## 🧠 Model Used

- Algorithm: Random Forest
- Preprocessing: StandardScaler
- Output: Binary classification (0 = No disease, 1 = Disease)

---

## 📊 Input Features

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise Induced Angina
- Oldpeak
- Slope
- CA
- Thal

---

## 🚀 Future Improvements

- Try XGBoost for better accuracy
- Deploy on Streamlit Cloud or Render
- Improve UI design
- Add explanation for predictions (AI explainability)

---

## 👩‍💻 Author

Built for learning and deployment practice using Python and Machine Learning.