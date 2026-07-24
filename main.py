import numpy as np
import pandas as pd
import os

'''
-----------------------------
Task 1: Data Cleaning Process
-----------------------------
'''

# reading the data from the csv file
curdir = os.path.dirname(__file__)
filepath = os.path.join(curdir, "Scenario 1 dataset.csv")
data = pd.read_csv(filepath, header=None, skiprows=1)
data = data.iloc[:, 1:]

# check for missing data
missing_values = data.isnull().sum()
is_missing = False

for i, val in enumerate(missing_values):
    if val != 0:
        print(f"Column {i+1} has {val} missing values")
        is_missing = True

complete_data = data.copy()

if is_missing:
    print("Filling missing values with column mean....")
    for col in complete_data.columns:
        # replace any instance of NaN by 'complete_data[col].mean()'
        complete_data[col] = complete_data[col].fillna(
            complete_data[col].mean()
        )

# check for outliers and handle them
for col in complete_data.columns:

    Q1 = complete_data[col].quantile(0.25)
    Q3 = complete_data[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = (
        (complete_data[col] < lower_bound)
        | (complete_data[col] > upper_bound)
    )

    if outliers.sum() != 0:
        print(f"Column {col}: {outliers.sum()} outliers detected")
        print("Replacing Outliers with Column median.....")

        # Replace outliers with column median
        median = complete_data[col].median()
        complete_data.loc[outliers, col] = median

'''
----------------------------
Task 2: Data Cleaning Process
-----------------------------
'''

# preforming the summary statistics
c1t_mean = complete_data[1].mean()
c1t_median = complete_data[1].median()
c1t_std = complete_data[1].std()

c2t_mean = complete_data[2].mean()
c2t_median = complete_data[2].median()
c2t_std = complete_data[2].std()

c3t_mean = complete_data[3].mean()
c3t_median = complete_data[3].median()
c3t_std = complete_data[3].std()

c1p_mean = complete_data[4].mean()
c1p_median = complete_data[4].median()
c1p_std = complete_data[4].std()

c2p_mean = complete_data[5].mean()
c2p_median = complete_data[5].median()
c2p_std = complete_data[5].std()

c3p_mean = complete_data[6].mean()
c3p_median = complete_data[6].median()
c3p_std = complete_data[6].std()

print("----- Summary Statistics -----")

print(
    f'1st city temperature: mean = {c1t_mean}, '
    f'median = {c1t_median}, '
    f'standard deviation = {c1t_std}, '
    f'percipatation: mean = {c1p_mean}, '
    f'median = {c1p_median}, '
    f'standard deviation = {c1p_std}'
)

print(
    f'2nd city temperature: mean = {c2t_mean}, '
    f'median = {c2t_median}, '
    f'standard deviation = {c2t_std}, '
    f'percipatation: mean = {c2p_mean}, '
    f'median = {c2p_median}, '
    f'standard deviation = {c2p_std}'
)

print(
    f'3rd city temperature: mean = {c3t_mean}, '
    f'median = {c3t_median}, '
    f'standard deviation = {c3t_std}, '
    f'percipatation: mean = {c3p_mean}, '
    f'median = {c3p_median}, '
    f'standard deviation = {c3p_std}'
)

# calculating the correlation coeff
correlation_matrix = complete_data.corr(method="pearson")

print("\n----- Correlation Coefficients -----")

city1_corr = correlation_matrix.iloc[0, 3]
print(f'1st city correlation coefficient = {city1_corr}')

city2_corr = correlation_matrix.iloc[1, 4]
print(f'2nd city correlation coefficient = {city2_corr}')

city3_corr = correlation_matrix.iloc[2, 5]
print(f'3rd city correlation coefficient = {city3_corr}')

'''
----------------------------
TASK 3 : data reprentation
----------------------------
'''

import matplotlib.pyplot as plt

cities = ['city1', 'city2', 'city3']

for i in range(3):

    plt.subplot(6, 1, 2 * i + 1)
    print(f'Plotting data for {cities[i]}')

    # teperature plots
    plt.plot(complete_data[i + 1], '-r', label='temperature')
    plt.ylabel('Temperature(°c)')
    plt.title(f'temp for city {cities[i]}')
    plt.legend()
    plt.grid(True)

    # precipiation plot
    print(f'Plotting precipitation data for {cities[i]}')

    plt.subplot(6, 1, 2 * i + 2)
    plt.plot(complete_data[i + 4], '-b', label='precipitation')
    plt.title(f'{cities[i]} precipitation')
    plt.grid(True)

plt.tight_layout()
plt.show()

# comarative temp plot
print("Plotting comparative temperature plot across cities")

plt.plot(complete_data[1], '-r', label='city1')
plt.plot(complete_data[2], '-b', label='city2')
plt.plot(complete_data[3], '-g', label='city3')

plt.title('comarative temp plot across cities')
plt.xlabel('days')
plt.ylabel('temperature')
plt.legend()
plt.grid(True)
plt.show()

'''
---------------------
TASK 4: Prediction
---------------------
Polynomial regression using temperature to predict precipitation
'''

# Predictor (x): temperature of city 1
x = complete_data[1].values

# Target (y): precipitation of city 1
y = complete_data[4].values

degree = 2  # you can try 1, 2, 3 and choose best based on metrics

coeffs = np.polyfit(x, y, degree)   # fit polynomial
model = np.poly1d(coeffs)           # build polynomial function

y_pred = model(x)

# Mean Squared Error (MSE)
mse = np.mean((y - y_pred) ** 2)

# Mean Absolute Error (MAE)
mae = np.mean(np.abs(y - y_pred))

# R-squared (R²)
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - (ss_res / ss_tot)

print("\n--- Task 4: Polynomial Regression (City 1) ---")
print("Polynomial degree:", degree)
print("Model coefficients:", coeffs)
print("MSE:", mse)
print("MAE:", mae)
print("R-squared:", r2)

import matplotlib.pyplot as plt

# Sort x for a smooth curve
idx = np.argsort(x)
x_sorted = x[idx]
y_sorted = y[idx]
y_fit_sorted = model(x_sorted)

plt.scatter(x, y, label="Actual data")
plt.plot(x_sorted, y_fit_sorted, label=f"Poly fit (deg={degree})")

plt.title("City 1: Precipitation vs Temperature (Polynomial Regression)")
plt.xlabel("Temperature (°C)")
plt.ylabel("Precipitation")
plt.grid(True)
plt.legend()
plt.show()