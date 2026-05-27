# RoadAccident - Accident Severity AI 🚗💥

An AI-powered web application that predicts the severity of road accidents based on various environmental and situational factors. 

## 🌟 Features
- **Accident Severity Prediction**: Uses a machine learning model (XGBoost/Random Forest) to predict whether an accident will result in slight, serious, or fatal injuries.
- **User Authentication**: Secure login and registration system for users.
- **Prediction History**: Logged-in users can view their past predictions.
- **Data Analytics Dashboard**: Visualizations of feature importance and model accuracy metrics.
- **Admin Panel**: Dedicated panel for administrators to view overall statistics and user data.

## 🛠️ Tech Stack
- **Backend Framework**: Python (Flask)
- **Database**: SQLite (SQLAlchemy)
- **Machine Learning**: Scikit-Learn, Pandas, Numpy, XGBoost
- **Frontend**: HTML, CSS, JavaScript (Chart.js for visualizations)

## 🚀 How to Run Locally

### 1. Setup a Virtual Environment (Recommended)
First, create and activate a Python virtual environment:
```powershell
python -m venv venv

# Activate on Windows:
.\venv\Scripts\Activate.ps1

# Activate on Mac/Linux:
source venv/bin/activate
```

### 2. Install Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 3. Run the Application
Start the Flask server:
```bash
python app.py
```
The application will be accessible in your web browser at `http://127.0.0.1:5000`.

## 🧠 Model Details
The AI uses a dataset of road accidents to learn patterns based on inputs such as:
- Day of the week
- Age band and sex of the driver
- Driving experience
- Weather and light conditions
- Road surface conditions
- Type of collision

The `model_comparison.py` script was used to train and compare Logistic Regression, Random Forest, and XGBoost models. The best-performing model is exported as `model.pkl` to serve live predictions.
