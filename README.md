# Climate-Data-Predictive-Modeling
## Overview
This repository contains a data analysis and machine learning project focused on forecasting precipitation levels across three distinct cities. By analyzing a 5-year meteorological dataset, the project explores multi-year weather trends, seasonal variations, and temperature-precipitation distributions, culminating in a predictive Polynomial Regression model for future rainfall estimation.

## Key Features
* **Data Cleaning & Preprocessing:** Handled missing values, standardized timestamps, and processed multi-variable weather records over a 5-year timeline using `pandas` and `NumPy`.
* **Exploratory Data Analysis (EDA):** Designed multi-line charts and distribution histograms to compare temperature fluctuations and rainfall patterns across three target cities.
* **Predictive Modeling:** Implemented a Polynomial Regression model to capture non-linear trend lines and predict future precipitation levels based on historical climate data.
* **Actionable Insights:** Extracted key seasonal patterns that informed parameter selection and model tuning for optimal forecasting performance.

## Tech Stack & Libraries
* **Language:** Python 3.x
* **Data Manipulation:** `pandas`, `NumPy`
* **Data Visualization:** `Matplotlib`, `Seaborn`
* **Machine Learning:** `scikit-learn` (PolynomialFeatures, LinearRegression)

## Project Structure
```text
├── data/
│   └── climate_data_5yr.csv        # Raw and cleaned meteorological dataset
├── notebooks/
│   └── climate_analysis.ipynb      # Interactive Jupyter Notebook with full analysis
├── src/
│   ├── data_preprocessing.py       # Data cleaning scripts
│   ├── visualization.py            # Plotting functions for EDA
│   └── regression_model.py         # Polynomial regression pipeline
├── outputs/
│   └── figures/                    # Generated histograms and line charts
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
Workflow & Methodology1. Data Preprocessing & CleaningNormalized raw multi-city meteorological parameters.Filtered outliers and filled missing entries using temporal interpolation.2. Visualization & InsightsBuilt multi-line time series plots showing cross-city temperature dynamics.Generated comparative histograms

Author
Musab Jamaleldien Mohamed Yousif
highlighting rainfall frequency and volume distributions.3. Predictive ModelingTo account for non-linear seasonal climate shifts, a Polynomial Regression approach was applied:Evaluated polynomial degrees to strike the optimal balance between bias and variance.Assessed performance using Root Mean Squared Error (RMSE) and $R^2$ metrics.
