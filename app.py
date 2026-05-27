from flask import Flask, render_template, request, redirect, url_for,session
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os

from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash


app = Flask(__name__)

app.config["SECRET_KEY"] = "secret123"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"

db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


# ==============================
# Load ML Model
# ==============================

# ==============================
# Load ML Model
# ==============================

model_package = pickle.load(open("model.pkl", "rb"))

model = model_package["model"]
preprocessor = model_package["preprocessor"]

label_encoder = pickle.load(open("label_encoder.pkl", "rb"))


# ==============================
# DATABASE MODELS
# ==============================

class User(UserMixin, db.Model):

    id = db.Column(db.Integer, primary_key=True)

    username = db.Column(db.String(50), unique=True, nullable=False)

    password = db.Column(db.String(200), nullable=False)

    is_admin = db.Column(db.Boolean, default=False)


class Prediction(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(db.Integer, nullable=False)

    severity = db.Column(db.String(50))

    confidence = db.Column(db.Float)

    day = db.Column(db.String(20))

    weather = db.Column(db.String(50))


# ==============================
# LOGIN MANAGER
# ==============================

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ==============================
# HOME
# ==============================

@app.route("/")
@login_required
def home():
    return render_template("index.html")


# ==============================
# REGISTER
# ==============================

@app.route("/register", methods=["GET","POST"])
def register():

    if request.method == "POST":

        username = request.form["username"]

        password = generate_password_hash(request.form["password"])

        user = User(username=username, password=password)

        db.session.add(user)

        db.session.commit()

        return redirect("/login")

    return render_template("register.html")

@app.route("/admin")
@login_required
def admin():

    if not current_user.is_admin:
        return "Access Denied"

    users = User.query.all()
    predictions = Prediction.query.all()

    total_users = len(users)
    total_predictions = len(predictions)

    # Severity analytics
    severity_counts = {}

    for p in predictions:
        severity_counts[p.severity] = severity_counts.get(p.severity, 0) + 1

    # Convert dict → lists for Chart.js
    severity_labels = list(severity_counts.keys())
    severity_values = list(severity_counts.values())

    return render_template(
        "admin.html",
        users=users,
        predictions=predictions,
        total_users=total_users,
        total_predictions=total_predictions,
        severity_labels=severity_labels,
        severity_values=severity_values
    )

@app.route("/analytics")
def analytics():

    predictions = Prediction.query.all()

    risks = [p.severity for p in predictions]

    return render_template(
        "analytics.html",
        risks=risks
    )

from flask_login import login_required, current_user

@app.route("/my_analytics")
@login_required
def my_analytics():
    # Fetch all predictions of the current user
    user_preds = Prediction.query.filter_by(user_id=current_user.id).all()

    # If user has no predictions, send empty list
    risks = [p.severity for p in user_preds] if user_preds else []

    return render_template("user_analytics.html", risks=risks)


# ==============================
# LOGIN
# ==============================

@app.route("/login", methods=["GET","POST"])
def login():

    if request.method == "POST":

        username = request.form["username"]

        password = request.form["password"]

        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):

            login_user(user)

            return redirect("/")

    return render_template("login.html")


# ==============================
# LOGOUT
# ==============================

@app.route("/logout")
@login_required
def logout():

    logout_user()

    return redirect("/login")

# ==============================
# HISTORY
# ==============================

@app.route("/history")
@login_required
def history():

    predictions = Prediction.query.filter_by(user_id=current_user.id).all()

    return render_template("history.html", predictions=predictions)


# ==============================
# CLEAR HISTORY
# ==============================

@app.route("/clear_history", methods=["POST"])
@login_required
def clear_history():

    Prediction.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()

    return redirect(url_for("history"))


# ==============================
# PREDICT
# ==============================

@app.route("/predict", methods=["POST"])
@login_required
def predict():

    input_data = {

        "Day_of_week": request.form.get("Day_of_week"),
        "Age_band_of_driver": request.form.get("Age_band_of_driver"),
        "Sex_of_driver": request.form.get("Sex_of_driver"),
        "Driving_experience": request.form.get("Driving_experience"),
        "Weather_conditions": request.form.get("Weather_conditions"),
        "Road_surface_conditions": request.form.get("Road_surface_conditions"),
        "Light_conditions": request.form.get("Light_conditions"),
        "Type_of_collision": request.form.get("Type_of_collision")

    }

    if None in input_data.values() or "" in input_data.values():

        return render_template(
            "index.html",
            error="⚠ Please fill all fields before prediction."
        )

    df = pd.DataFrame([input_data])

# Preprocess input
    X_processed = preprocessor.transform(df)

    prediction_encoded = model.predict(X_processed)[0]

    prediction = label_encoder.inverse_transform([prediction_encoded])[0]

    probabilities = [0,0,0]
    confidence = 0

    try:

        probs = model.predict_proba(X_processed)[0]

        prob_dict = {
            label_encoder.classes_[i]: float(probs[i])
            for i in range(len(probs))
        }

        probabilities = [
            prob_dict.get("Slight Injury",0),
            prob_dict.get("Serious Injury",0),
            prob_dict.get("Fatal Injury",0)
        ]

        confidence = round(max(probabilities) * 100,2)

    except Exception as e:
        print("Probability error:", e)
    # SAVE PREDICTION

    print("Model classes:", label_encoder.classes_)
    print("Probabilities:", probs)

    new_prediction = Prediction(

        user_id=current_user.id,

        severity=prediction,

        confidence=confidence,

        day=input_data["Day_of_week"],

        weather=input_data["Weather_conditions"]

    )

    db.session.add(new_prediction)

    db.session.commit()

    return render_template(

        "index.html",

        prediction_text=prediction,

        confidence=confidence,

        probabilities=probabilities

    )



# ==============================
# DASHBOARD
# ==============================

@app.route("/dashboard")
@login_required
def dashboard():

    results = {}

    if os.path.exists("model_results.json"):

        with open("model_results.json") as f:

            results = json.load(f)

    feature_plot_available = False

    try:

        classifier = model.named_steps["classifier"]

        if hasattr(classifier, "feature_importances_"):

            importances = classifier.feature_importances_

            ohe = model.named_steps["preprocessor"] \
                .named_transformers_["cat"] \
                .named_steps["encoder"]

            feature_names = ohe.get_feature_names_out()

            indices = np.argsort(importances)[-10:]

            plt.figure(figsize=(8,6))

            plt.barh(range(len(indices)), importances[indices])

            plt.yticks(range(len(indices)), feature_names[indices])

            plt.title("Top 10 Feature Importance")

            plt.tight_layout()

            if not os.path.exists("static"):

                os.makedirs("static")

            plt.savefig("static/feature_importance.png")

            plt.close()

            feature_plot_available = True

    except Exception as e:

        print("Feature importance error:", e)

        feature_plot_available = False

    return render_template(

        "dashboard.html",

        results=results,

        feature_plot_available=feature_plot_available

    )


# ==============================
# CREATE DATABASE + RUN APP
# ==============================

if __name__ == "__main__":

    with app.app_context():

        db.create_all()

    app.run(debug=True)

    #app.run(host="0.0.0.0", port=5000)
