
# 🧠🔬 Multiple Disease Prediction System using Machine Learning and Streamlit

This project is a **Multiple Disease Prediction Web Application** that helps users detect two common health issues — **Diabetes** and **Heart Disease** — using **Machine Learning models**. The web app is built using **Streamlit** and deployed on **Streamlit Cloud**.

---

## 📌 Project Features

- Two Machine Learning models:
  - **Diabetes Prediction using SVM (Support Vector Machine)**
  - **Heart Disease Prediction using Logistic Regression**
- Clean and interactive UI built using **Streamlit**
- Models built and trained in **Jupyter Notebook**
- Web App developed in **PyCharm**
- Deployment on **Streamlit Cloud**
- Models saved using **Pickle** for production usage

---

## 🧪 Model 1: Diabetes Prediction using SVM

### 🔍 Dataset Used
- Source: `diabetes.csv`
- Shape: `(768, 9)`
- Target column: `Outcome` (0: Non-Diabetic, 1: Diabetic)

### 📈 Input Features (User Parameters)
1. Pregnancies
2. Glucose
3. BloodPressure
4. SkinThickness
5. Insulin
6. BMI
7. DiabetesPedigreeFunction
8. Age

### 🧹 Data Preprocessing
- Standardization using `StandardScaler`
- Splitting into training and test sets using `train_test_split`
- 80% training, 20% test data with stratified sampling

```python
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)
```

### 🤖 Model Training
- Model: `SVC(kernel='linear')` from `sklearn.svm`
- Evaluation using `accuracy_score`

```python
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)
```

---

## ❤️ Model 2: Heart Disease Prediction using Logistic Regression

### 🔍 Dataset Used
- Source: `heart_disease_data.csv`
- Shape: `(303, 14)`
- Target column: `target` (0: No Disease, 1: Has Disease)

### 📈 Input Features (User Parameters)
1. age
2. sex
3. cp (chest pain type)
4. trestbps (resting blood pressure)
5. chol (serum cholesterol)
6. fbs (fasting blood sugar)
7. restecg (resting ECG)
8. thalach (max heart rate)
9. exang (exercise-induced angina)
10. oldpeak (ST depression)
11. slope
12. ca (major vessels colored)
13. thal (defect type)

### 🧹 Data Preprocessing
- Missing values check
- Label separation (`X` and `Y`)
- Train-test split using `train_test_split`

### 🤖 Model Training
- Model: `LogisticRegression()` from `sklearn.linear_model`
- Evaluation using `accuracy_score`

```python
model = LogisticRegression()
model.fit(X_train, Y_train)
```

---

## 💻 Web App Development with Streamlit

The entire frontend is built using **Streamlit**, a lightweight Python framework for creating ML web apps.

### 📂 Project Structure (PyCharm)
```
📁 project-folder/
│
├── diabetes_model.pkl
├── heart_disease_model.pkl
├── app.py
├── requirements.txt
└── README.md
```

### 🧠 Streamlit Logic

- Sidebar Navigation using `streamlit_option_menu`
- User input forms in 3-column layout
- Inputs processed and passed to the model
- Predictions displayed using `st.success()`

Example:
```python
if st.button('Diabetes Test Result'):
    user_input = [float(Pregnancies), float(Glucose), ...]
    prediction = diabetes_model.predict([user_input])
    ...
```

---

## ☁️ Deployment on Streamlit Cloud

### 🌐 Steps to Deploy:
1. Push your code to a public GitHub repository
2. Visit: [https://share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select the repository and main Python file (`app.py`)
5. Click “Deploy”

✅ Your app is now live on the web!

live web app link: [https://multi-disease-prediction-2xjfl9oq7tm7cmv68g27px.streamlit.app/]
---

## 🔧 Tech Stack

- Python
- Pandas, NumPy, Scikit-Learn
- SVM, Logistic Regression
- Pickle (model serialization)
- Streamlit (frontend + backend)
- PyCharm (IDE)
- Jupyter Notebook (model building)

---

## 📚 Future Improvements

- Add more disease prediction models (e.g., kidney disease, liver disease)
- Add visual analytics (charts and plots)
- Include user authentication and history tracking

---

## 📬 Contact

For any queries or suggestions:

**Abhay**  
Email: [abhaykumarchandra422@gmail.com]  
GitHub: [https://github.com/abhaykumar15]  

---
