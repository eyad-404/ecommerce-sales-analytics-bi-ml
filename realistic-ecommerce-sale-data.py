import pandas as pd

df = pd.read_csv("Realistic_E_Commerce_Sales_Data.csv")

"""# **Data Cleaning**"""

df["Customer ID"] = df["Customer ID"].astype(str)

df["Gender"] = df["Gender"].astype(str)

df["Region"] = df["Region"].astype(str)

df["Product Name"] = df["Product Name"].astype(str)

df["Category"] = df["Category"].astype(str)

df["Shipping Status"] = df["Shipping Status"].astype(str)

df["Age"] = pd.to_numeric(
    df["Age"],
    errors="coerce"
)

df["Unit Price"] = pd.to_numeric(
    df["Unit Price"],
    errors="coerce"
)

df["Quantity"] = pd.to_numeric(
    df["Quantity"],
    errors="coerce"
)

df["Total Price"] = pd.to_numeric(
    df["Total Price"],
    errors="coerce"
)

df["Shipping Fee"] = pd.to_numeric(
    df["Shipping Fee"],
    errors="coerce"
)

df["Region"] = df["Region"].replace(
    "",
    "Unknown"
)

df["Shipping Status"] = df["Shipping Status"].replace(
    "",
    "Unknown"
)

df["Age"] = df["Age"].fillna(0)

df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    dayfirst=True
)

def age_group(age):

    if age <= 25:
        return "18-25"

    elif age <= 35:
        return "26-35"

    elif age <= 45:
        return "36-45"

    elif age <= 55:
        return "46-55"

    else:
        return 55

df["Age Group"] = df["Age"].apply(
    age_group
)

df["Month Name"] = df["Order Date"].dt.month_name()



df["Year"] = df["Order Date"].dt.year

print(df.head())

print(df.info())

"""# **Dax Measures**"""

# total_orders

total_orders = len(df)

print(total_orders)

# returned_orders

returned_orders = df[
    df["Shipping Status"] == "Returned"
].shape[0]

print(returned_orders)

# return_rate

if total_orders != 0:

    return_rate = (
        returned_orders / total_orders
    ) * 100

else:

    return_rate = 0


print(return_rate)

total_quantity_sold = df["Quantity"].sum()

print(total_quantity_sold)

total_revenue = df["Total Price"].sum()

print(total_revenue)

"""# Data **Preprocessing**"""

X = df[[
    "Region",
    "Category",
    "Quantity",
    "Shipping Status",
    "Age Group",
    "Month Name"
]]

y = df["Total Price"]

X_encoded = pd.get_dummies(X, drop_first=True)

X_encoded.head()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

"""# **Linear Regression**"""

from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()

lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np

lr_mae = mean_absolute_error(y_test, lr_predictions)

lr_mse = mean_squared_error(y_test, lr_predictions)

lr_rmse = np.sqrt(lr_mse)

lr_r2 = r2_score(y_test, lr_predictions)

print("Linear Regression Results")
print("MAE:", lr_mae)
print("MSE:", lr_mse)
print("RMSE:", lr_rmse)
print("R² Score:", lr_r2)

"""# **Random Forest**"""

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators=100,
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_predictions)

rf_mse = mean_squared_error(y_test, rf_predictions)

rf_rmse = np.sqrt(rf_mse)

rf_r2 = r2_score(y_test, rf_predictions)

print("Random Forest Results")
print("MAE:", rf_mae)
print("MSE:", rf_mse)
print("RMSE:", rf_rmse)
print("R² Score:", rf_r2)

comparison = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'MAE': [lr_mae, rf_mae],
    'MSE': [lr_mse, rf_mse],
    'RMSE': [lr_rmse, rf_rmse],
    'R² Score': [lr_r2, rf_r2]
})

comparison

"""# **Actual vs Predicted Visualization**"""

import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))

plt.scatter(y_test, lr_predictions)

plt.xlabel("Actual Revenue")

plt.ylabel("Predicted Revenue")

plt.title("Linear Regression: Actual vs Predicted")

plt.show()

plt.figure(figsize=(8,6))

plt.scatter(y_test, rf_predictions)

plt.xlabel("Actual Revenue")

plt.ylabel("Predicted Revenue")

plt.title("Random Forest: Actual vs Predicted")

plt.show()

feature_importance = pd.DataFrame({
    'Feature': X_encoded.columns,
    'Importance': rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

feature_importance.head(10)

plt.figure(figsize=(10,6))

plt.barh(
    feature_importance['Feature'].head(10),
    feature_importance['Importance'].head(10)
)

plt.xlabel("Importance")

plt.ylabel("Feature")

plt.title("Top 10 Important Features")

plt.gca().invert_yaxis()

plt.show()
