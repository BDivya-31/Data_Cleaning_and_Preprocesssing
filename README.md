# Data_Cleaning_and_Preprocesssing

# Task 1: Data Cleaning & Preprocessing

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=flat&logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?style=flat&logo=numpy)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=flat&logo=scikit-learn)

---

## 📌 Overview

This repository contains the solution for **Task 1: Data Cleaning & Preprocessing**. Raw real-world datasets often contain missing values, noisy features, outliers, and inconsistent data types that degrade machine learning model performance. 

The primary objective of this project is to execute an end-to-end data cleaning pipeline to transform raw tabular data into a refined, normalized, and machine-readable format ready for ML algorithm ingestion.

---

## 📊 Dataset Description

* **Source:** Travel & Tourism Dataset
* **Raw Data Dimensions:** `1000` rows $\times$ `29` columns
* **Target Output:** `cleaned_travel_tourism_dataset.csv`

The dataset consists of demographic, operational, and financial attributes related to travel bookings, customer profiles, and trip metrics.
This Dataset abstracted from Kaggle, Here is the link of Dataset:
https://www.kaggle.com/datasets/madhavw/travel-and-tourism

---

## 🛠️ Step-by-Step Preprocessing Pipeline

The data preparation workflow follows a structured 5-step pipeline:
### 1. Data Exploration & Quality Audit
- Inspected overall dataset dimensions (`df.shape`), column data types (`df.dtypes`), and initial statistical summaries (`df.describe()`).
- Identified non-null counts and calculated the percentage of missing values per feature using `.isnull().sum()`.

### 2. Missing Value Imputation
- **Numerical Features:** Imputed null values using the column **median** to prevent skewing caused by non-normal distributions.
- **Categorical Features:** Imputed missing entries using the column **mode** (most frequent category).

### 3. Categorical Feature Encoding
- **Label Encoding (`LabelEncoder`):** Applied to binary and ordinal categorical features to convert labels into sequential numerical integers.
- **One-Hot Encoding (`pd.get_dummies`):** Applied to nominal multi-class features without inherent ordering, avoiding artificial ordinal bias while dropping the first column to prevent the dummy variable trap.

### 4. Outlier Detection & Removal
- Visualized numerical distributions and feature spreads using Seaborn **Boxplots**.
- Applied the **Interquartile Range (IQR) Filter** to remove extreme outliers:
  $$\text{IQR} = Q_3 - Q_1$$
  $$\text{Lower Bound} = Q_1 - 1.5 \times \text{IQR}$$
  $$\text{Upper Bound} = Q_3 + 1.5 \times \text{IQR}$$

### 5. Feature Normalization & Standardization
- Applied **StandardScaler** (Z-Score Normalization) to center numerical features to mean zero ($\mu = 0$) and standard deviation one ($\sigma = 1$):
  $$x_{\text{scaled}} = \frac{x - \mu}{\sigma}$$
- Ensures gradient descent convergence and prevents high-magnitude features from dominating model training.
