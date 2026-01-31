# 🩺 Diabetes Prediction using Machine Learning

## 📌 Project Overview

This project predicts whether a person is **diabetic or non-diabetic** based on medical attributes such as glucose level, blood pressure, BMI, and age. It is a **beginner-friendly machine learning classification project** that demonstrates the complete ML pipeline and includes a **Streamlit web application** for real-time predictions.

---

## 🎯 Objective

To build a **binary classification model** that can predict diabetes risk using patient health data.

---

## 📊 Dataset

* **Source:** Kaggle (Pima Indians Diabetes Dataset)
* **Link:** [https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

### Features

* Pregnancies
* Glucose
* BloodPressure
* SkinThickness
* Insulin
* BMI
* DiabetesPedigreeFunction
* Age

### Target Variable

* `Outcome`

  * 1 → Diabetic
  * 0 → Non-Diabetic

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib / Seaborn
* Scikit-learn
* Streamlit

---

## 🧠 Machine Learning Algorithms

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Random Forest Classifier

---

## 🔄 Project Workflow

1. Import required libraries
2. Load and explore the dataset
3. Handle missing or invalid values
4. Perform exploratory data analysis (EDA)
5. Feature scaling using StandardScaler
6. Split data into training and testing sets
7. Train classification models
8. Evaluate models using performance metrics
9. Save trained model
10. Deploy using Streamlit

---

## 📈 Model Evaluation Metrics

* Accuracy Score
* Precision
* Recall
* Confusion Matrix

---

## 🌐 Streamlit Web Application

The Streamlit app allows users to input medical information and receive instant diabetes predictions.

### App Features

* Simple and clean UI
* Real-time prediction
* Probability-based output

### ▶️ Run Streamlit App

```
pip install -r requirements.txt
streamlit run app.py
```

---

## 📂 Project Structure

```
Diabetes-Prediction/
│── data/
│   └── diabetes.csv
│── notebook/
│   └── diabetes_prediction.ipynb
│── app.py
│── scaled.pkl
│── model.pkl
│── requirements.txt
│── README.md
```

---

## 🚀 How to Run the Project

1. Clone the repository

```
git clone https://github.com/sumitkumar1233edeedad/-Diabetes-Prediction-using-Machine-Learning.git
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Run the Streamlit app

```
streamlit run app.py
```

---

## 📌 Results

The model predicts diabetes risk with good accuracy on unseen data. Performance can be further improved using advanced models and hyperparameter tuning.

---

## 🌱 Future Improvements

* Hyperparameter tuning
* Use advanced models like XGBoost
* Deploy on Streamlit Cloud
* Add more health indicators

---

## 👤 Author

**Vanshuu**

---

## ⭐ Acknowledgment

Dataset provided by Kaggle and the UCI Machine Learning Repository.

---

