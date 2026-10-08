import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

df = pd.read_csv("player_data.csv", encoding="latin1")

# Handle missing numeric values (fill NA with 0 or column median)
X = df.select_dtypes(include=["number"]).drop(columns=["Salary"], errors="ignore")
X = X.fillna(0)
y = df["Salary"]

# Group Center, Wingers, and slash combinations into 'FWD'
if "Position" in df.columns:
    df["Position"] = df["Position"].astype(str).str.upper().str.strip()
    fwd_pattern = r"^(?:(?:C|RW|LW)/)*(?:C|RW|LW)(?:/(?:C|RW|LW))*$"
    df.loc[df["Position"].str.contains(fwd_pattern, regex=True, na=False), "Position"] = "FWD"

# Convert categorical features (e.g., Position) into one-hot encoded variables
if "Position" in df.columns:
    pos_dummies = pd.get_dummies(df["Position"], prefix="Pos", drop_first=True)
    X = pd.concat([X, pos_dummies], axis=1)

# Train and Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = RandomForestRegressor(
    n_estimators=100, random_state=42
)
model.fit(X_train, y_train)


predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

# Cross-Validation Score across the full dataset for stability check
cv_r2 = cross_val_score(model, X, y, cv=5, scoring="r2").mean()


print(f"Test R-squared (R²): {r2:.3f}")

importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(
    ascending=False
)
print("--- TOP 10 FEATURE IMPORTANCES ---")
print(importances.head(10))

# Plot 1: Feature Importances
plt.figure(figsize=(10, 5))
sns.barplot(x=importances.head(10).values, y=importances.head(10).index, palette="viridis")
plt.title("Top 10 Drivers of NHL Salary")
plt.xlabel("Importance Score")
plt.ylabel("Stat / Feature")
plt.tight_layout()
plt.show()

# Plot 2: Actual vs. Predicted Salaries
plt.figure(figsize=(8, 6))
plt.scatter(y_test / 1e6, predictions / 1e6, alpha=0.7, color="navy")
plt.plot([0, y_test.max()/1e6], [0, y_test.max()/1e6], "r--", label="Perfect Prediction")
plt.title("Actual vs. Predicted Salary ($ Millions)")
plt.xlabel("Actual Salary ($M)")
plt.ylabel("Predicted Salary ($M)")
plt.legend()
plt.tight_layout()
plt.show()