import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# File path
DATA_PATH = "Salary Data.csv"

# Load dataset
data = pd.read_csv(DATA_PATH)
print("First few rows of data:")
print(data.head())

# Drop missing values
data = data.dropna(subset=["Salary"])

# Features and target
X = data.drop(columns=["Salary"])
y = data["Salary"]

# Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include=["object"]).columns
numeric_cols = X.select_dtypes(exclude=["object"]).columns

print("\nCategorical columns:", list(categorical_cols))
print("Numeric columns:", list(numeric_cols))

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

# Full model pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model.fit(X_train, y_train)

# Evaluate model
preds = model.predict(X_test)
#rmse = mean_squared_error(y_test, preds, squared=False)
rmse = np.sqrt(mean_squared_error(y_test, preds))

r2 = r2_score(y_test, preds)

print(f"\nModel Performance:")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/salary_model.pkl")
print("\nModel saved successfully as models/salary_model.pkl ✅")



