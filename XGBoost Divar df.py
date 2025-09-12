import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

df = pd.read_excel("clean_divar_dataset.xlsx")

non_numeric_cols = df.select_dtypes(include=["object"]).columns
df = df.drop(columns=non_numeric_cols)

X = df.drop(columns=["rent_value"])
y = np.log1p(df["rent_value"])   # 🔹 log(1 + rent_value)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method="hist"
)

model.fit(X_train, y_train)

y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)   # 🔹 exp(log+1) - 1

y_test_true = np.expm1(y_test)

mae = mean_absolute_error(y_test_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_true, y_pred))
r2 = r2_score(y_test_true, y_pred)
accuracy = 100 * (1 - mae / y_test_true.mean())

print("📊 Evaluation Metrics with Log Transform:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")
print(f"Custom Accuracy: {accuracy:.2f}%")
