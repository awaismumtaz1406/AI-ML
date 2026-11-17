
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. CREATE DATASET

data = {
    "area": [
        800, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300,
        1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800,
        1850, 1900, 1950, 2000, 2050, 2100, 2200, 2300, 2400, 2500
    ],
    "bedrooms": [
        2, 2, 2, 2, 3, 2, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 4, 3, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 5, 5
    ],
    "bathrooms": [
        1, 2, 1, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 3, 2, 3, 3, 2, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 4, 4
    ],
    "location": [
        "Suburban", "Suburban", "Urban", "Suburban", "Urban",
        "Suburban", "Urban", "Suburban", "Urban", "Suburban",
        "Urban", "Suburban", "Urban", "Suburban", "Urban",
        "Suburban", "Urban", "Urban", "Suburban", "Urban",
        "Suburban", "Urban", "Suburban", "Urban", "Suburban",
        "Urban", "Suburban", "Urban", "Urban", "Suburban"
    ],
    "price": [
        4500000, 4900000, 5200000, 5400000, 5900000,
        5700000, 6300000, 6500000, 6800000, 7000000,
        7300000, 7500000, 7800000, 8000000, 8500000,
        8300000, 8900000, 9400000, 9000000, 9900000,
        10200000, 10500000, 10800000, 11200000, 11500000,
        11800000, 12200000, 12700000, 13500000, 14000000
    ]
}

df = pd.DataFrame(data)

# Add missing value
df.loc[3, "bathrooms"] = np.nan

# Add an exact duplicate
df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

# Add an unrealistic price outlier
outlier_row = pd.DataFrame({
    "area": [800],
    "bedrooms": [2],
    "bathrooms": [1],
    "location": ["Suburban"],
    "price": [50000000]
})

df = pd.concat([df, outlier_row], ignore_index=True)

print("Dataset Source: Self-created synthetic dataset")

# 2. DATASET INSPECTION

print("\n--- Dataset Inspection ---")
print(df.head())
print("\nShape:", df.shape)
print("\nData Types:")
print(df.dtypes)
print("\nDescriptive Statistics:")
print(df.describe(include="all"))

# 3. DATA CLEANING

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

print("\nDuplicate Rows Before Cleaning:")
print(df.duplicated().sum())

# Fill missing numerical values using median
df["bathrooms"] = df["bathrooms"].fillna(
    df["bathrooms"].median()
)

# Remove duplicate rows
df = df.drop_duplicates().copy()

# Detect outliers using IQR
Q1 = df["price"].quantile(0.25)
Q3 = df["price"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = df[
    (df["price"] < lower_limit) |
    (df["price"] > upper_limit)
]

print("\nDetected Outliers:")
print(outliers)

# Remove extreme price outliers
df = df[
    (df["price"] >= lower_limit) &
    (df["price"] <= upper_limit)
].copy()

print("\n--- After Cleaning ---")
print("New Shape:", df.shape)
print("Missing Values:")
print(df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

# 4. FEATURE AND PRICE RELATIONSHIP

plt.figure(figsize=(8, 5))

plt.scatter(df["area"], df["price"])

plt.title("Area vs House Price")
plt.xlabel("Area (Square Feet)")
plt.ylabel("Price (PKR)")
plt.grid(True)

plt.tight_layout()
plt.show()

# 5. FEATURE SELECTION

X = df[
    ["area", "bedrooms", "bathrooms", "location"]
]

y = df["price"]

numeric_features = [
    "area", "bedrooms", "bathrooms"
]

categorical_features = ["location"]

# 6. TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

# 7. PREPROCESSING

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features
        ),
        (
            "location",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

# 8. LINEAR REGRESSION MODEL

model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("regression", LinearRegression())
    ]
)

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# 9. COEFFICIENTS AND INTERCEPT

feature_names = (
    model.named_steps["preprocessing"]
    .get_feature_names_out()
)

coefficients = model.named_steps["regression"].coef_

intercept = model.named_steps["regression"].intercept_

print("\n--- Model Coefficients ---")

for name, value in zip(feature_names, coefficients):
    print(name, ":", round(value, 2))

print("\nIntercept:", round(intercept, 2))

# 10. TEST PREDICTIONS

y_pred = model.predict(X_test)

results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred,
    "Absolute Error": np.abs(y_test.values - y_pred)
})

print("\n--- Test Predictions ---")
print(results.round(2).to_string(index=False))

# 11. NEW HOUSE PREDICTION

new_house = pd.DataFrame({
    "area": [1800],
    "bedrooms": [4],
    "bathrooms": [3],
    "location": ["Urban"]
})

new_price = model.predict(new_house)[0]

print("\n--- New House Prediction ---")
print(new_house)

print(
    "Predicted Price: PKR",
    round(new_price, 2)
)

# 12. MODEL EVALUATION

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\n--- Evaluation Metrics ---")

print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# 13. ACTUAL VS PREDICTED GRAPH

plt.figure(figsize=(7, 5))

plt.scatter(y_test, y_pred)

minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.title("Actual vs Predicted House Prices")
plt.xlabel("Actual Price (PKR)")
plt.ylabel("Predicted Price (PKR)")
plt.grid(True)

plt.tight_layout()
plt.show()

# 14. PREDICTION ERROR TABLE

print("\n--- Prediction Error Table ---")

print(
    results.head(5).round(2).to_string(index=False)
)

# 15. INTERPRETATION

print("\n--- Model Interpretation ---")

print("MAE and RMSE show prediction error in PKR.")

print("R2 shows how much price variation is explained by the model.")

print(
    "Limitation: The dataset is small and synthetic. "
    "Real house prices also depend on property condition, "
    "exact location, age and market demand."
)
