# 🛒 Realistic E-Commerce Sales Data Analysis

A data science project that performs **end-to-end analysis and revenue prediction** on a realistic e-commerce sales dataset. The workflow covers data cleaning, feature engineering, key business KPI calculation, and machine learning model training/evaluation using **Linear Regression** and **Random Forest**.

---

## 📁 Project Structure

```
code/
├── realistic-ecommerce-sale-data.py   # Main analysis script
├── Realistic_E_Commerce_Sales_Data.csv     # Input dataset (required)
└── README.md
```

---

## 🔍 What the Script Does

### 1. 🧹 Data Cleaning
- Loads the dataset from `Realistic_E_Commerce_Sales_Data.csv`
- Casts categorical columns (`Customer ID`, `Gender`, `Region`, `Product Name`, `Category`, `Shipping Status`) to string
- Converts numerical columns (`Age`, `Unit Price`, `Quantity`, `Total Price`, `Shipping Fee`) with `coerce` error handling to deal with malformed values
- Replaces empty strings in `Region` and `Shipping Status` with `"Unknown"`
- Fills missing `Age` values with `0`
- Parses `Order Date` as a datetime (day-first format)

### 2. 🧠 Feature Engineering
| New Column   | Description                                          |
|--------------|------------------------------------------------------|
| `Age Group`  | Bins customer age into ranges: `18-25`, `26-35`, `36-45`, `46-55`, `55+` |
| `Month Name` | Extracts the month name from `Order Date`            |
| `Year`       | Extracts the year from `Order Date`                  |

### 3. 📊 DAX-Style KPI Measures
The script computes the following business metrics:

| Metric               | Description                                    |
|----------------------|------------------------------------------------|
| `total_orders`       | Total number of rows/orders                    |
| `returned_orders`    | Count of orders with `Shipping Status == "Returned"` |
| `return_rate`        | `(returned_orders / total_orders) * 100`       |
| `total_quantity_sold`| Sum of all quantities sold                     |
| `total_revenue`      | Sum of all `Total Price` values                |

### 4. 🤖 Machine Learning – Revenue Prediction

**Features used:**
- `Region`, `Category`, `Quantity`, `Shipping Status`, `Age Group`, `Month Name`
- One-hot encoded via `pd.get_dummies(drop_first=True)`

**Train/Test Split:** 80% / 20%

#### Models Trained:
| Model              | Library                          |
|--------------------|----------------------------------|
| Linear Regression  | `sklearn.linear_model`           |
| Random Forest      | `sklearn.ensemble` (100 trees)   |

**Evaluation Metrics:** MAE, MSE, RMSE, R² Score

### 5. 📈 Visualizations
- **Actual vs Predicted scatter plots** for both Linear Regression and Random Forest
- **Top 10 Feature Importance bar chart** from the Random Forest model

---

## 🛠️ Requirements

Install dependencies with:

```bash
pip install pandas scikit-learn matplotlib numpy
```

| Library       | Purpose                              |
|---------------|--------------------------------------|
| `pandas`      | Data loading, cleaning, engineering  |
| `numpy`       | Numerical operations (RMSE)          |
| `scikit-learn`| ML models, train/test split, metrics |
| `matplotlib`  | Data visualization                   |

---

## 🚀 How to Run

1. **Place the dataset** in the same directory as the script:
   ```
   Realistic_E_Commerce_Sales_Data.csv
   ```

2. **Run the script:**
   ```bash
   python "realistic_ecommerce_sales_data (1).py"
   ```

3. The script will print:
   - A preview of the cleaned dataframe (`df.head()`, `df.info()`)
   - All computed KPI values
   - Training/testing shape info
   - Evaluation metrics for both models
   - A side-by-side comparison table of model performance

4. Matplotlib windows will pop up showing the scatter plots and feature importance chart.

---

## 📋 Dataset Columns

The script expects a CSV with at least the following columns:

| Column           | Type        | Description                        |
|------------------|-------------|------------------------------------|
| `Customer ID`    | String      | Unique customer identifier         |
| `Gender`         | String      | Customer gender                    |
| `Age`            | Numeric     | Customer age                       |
| `Region`         | String      | Customer region                    |
| `Product Name`   | String      | Name of the product purchased      |
| `Category`       | String      | Product category                   |
| `Unit Price`     | Numeric     | Price per unit                     |
| `Quantity`       | Numeric     | Number of units ordered            |
| `Total Price`    | Numeric     | Total order value (target variable)|
| `Shipping Fee`   | Numeric     | Shipping cost                      |
| `Shipping Status`| String      | e.g., Delivered, Returned, Pending |
| `Order Date`     | Date (DD/MM/YYYY) | Date the order was placed    |

---

## 📌 Notes

- The script was originally developed as a **Google Colab notebook** and exported to `.py` format.
- The `age_group()` function returns the integer `55` (not the string `"55+"`) for ages above 55 — this is a minor inconsistency that can be fixed by changing `return 55` to `return "55+"`.
- No `random_state` is set in the `train_test_split`, so results may vary slightly between runs.

---

## 📄 License

This project is for educational and analytical purposes.
