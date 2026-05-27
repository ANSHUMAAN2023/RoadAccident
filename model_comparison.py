import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE

# ==============================
# Create static folder
# ==============================

if not os.path.exists("static"):
    os.makedirs("static")

# ==============================
# Load Dataset
# ==============================

df = pd.read_csv("dataset.csv")

# Drop Time column if exists
if "Time" in df.columns:
    df = df.drop("Time", axis=1)

# ==============================
# Encode Target
# ==============================

label_encoder = LabelEncoder()
df["Accident_severity"] = label_encoder.fit_transform(df["Accident_severity"])

y = df["Accident_severity"]

# ==============================
# Feature Selection (More Features)
# ==============================

possible_features = [
    'Day_of_week',
    'Age_band_of_driver',
    'Sex_of_driver',
    'Driving_experience',
    'Weather_conditions',
    'Road_surface_conditions',
    'Light_conditions',
    'Type_of_collision',
    'Road_type'
]

selected_features = [col for col in possible_features if col in df.columns]

print("Using features:", selected_features)

X = df[selected_features]

X = df[selected_features]

# ==============================
# Preprocessing
# ==============================

categorical_cols = X.select_dtypes(include="object").columns

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("cat", categorical_pipeline, categorical_cols)
])

# ==============================
# Train Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============================
# Transform Features
# ==============================

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# ==============================
# Handle Class Imbalance
# ==============================

smote = SMOTE(random_state=42)

X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train_processed,
    y_train
)

print("After SMOTE class distribution:")
print(pd.Series(y_train_resampled).value_counts())

# ==============================
# Define Models
# ==============================

models = {

    "Logistic_Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Random_Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        num_class=3,
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8
    )
}

results = {}
trained_models = {}

# ==============================
# Train Models
# ==============================

for name, model in models.items():

    print(f"\n🔹 Training {name}...")

    model.fit(X_train_resampled, y_train_resampled)

    y_pred = model.predict(X_test_processed)

    acc = accuracy_score(y_test, y_pred)

    results[name] = float(acc)

    trained_models[name] = model

    print(f"{name} Accuracy: {acc:.4f}")

    print("\nClassification Report:")

    print(classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    ))

    # ==============================
    # Confusion Matrix
    # ==============================

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5,4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )

    plt.title(f"{name} Confusion Matrix")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    plt.savefig(f"static/{name}_confusion_matrix.png")

    plt.close()

# ==============================
# Save Accuracy Results
# ==============================

with open("model_results.json","w") as f:
    json.dump(results,f,indent=4)

# ==============================
# Accuracy Graph
# ==============================

plt.figure(figsize=(6,4))

plt.bar(results.keys(),results.values())

plt.title("Model Accuracy Comparison")

plt.ylabel("Accuracy")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig("static/model_accuracy_comparison.png")

plt.close()

# ==============================
# Best Model
# ==============================

best_model_name = max(results,key=results.get)

print(f"\n🏆 Best Model: {best_model_name}")

best_model = trained_models[best_model_name]

# ==============================
# Save Model
# ==============================

model_package = {
    "model": best_model,
    "preprocessor": preprocessor
}

pickle.dump(model_package,open("model.pkl","wb"))
pickle.dump(label_encoder,open("label_encoder.pkl","wb"))

# ==============================
# Feature Importance (RF)
# ==============================

if "Random_Forest" in trained_models:

    rf_model = trained_models["Random_Forest"]

    ohe = preprocessor.named_transformers_["cat"] \
        .named_steps["encoder"]

    feature_names = ohe.get_feature_names_out(categorical_cols)

    importances = rf_model.feature_importances_

    indices = np.argsort(importances)[-20:]

    plt.figure(figsize=(10,8))

    plt.barh(range(len(indices)),importances[indices])

    plt.yticks(range(len(indices)),feature_names[indices])

    plt.title("Top 20 Feature Importance (Random Forest)")

    plt.tight_layout()

    plt.savefig("static/feature_importance.png")

    plt.close()

print("\n✅ Training Completed Successfully!")