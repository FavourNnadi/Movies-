import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import shapiro
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# Try reading the files with different encodings
encodings_to_try = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']

for encoding in encodings_to_try:
    try:
        movie1 = pd.read_csv('movie1.csv', encoding=encoding)
        movie2 = pd.read_csv('movie2.csv', encoding=encoding)
        break  # Stop trying encodings if successful
    except UnicodeDecodeError:
        continue  # Try the next encoding

# Merge the datasets based on 'Movie'/'Title' and 'Year' columns
merged_data = pd.merge(movie1, movie2, left_on=['Title', 'Year'], right_on=['Title', 'Year'], how='inner')

# If you want to save the merged data to a new CSV file
merged_data.to_csv('merged_movies.csv', index=False) 


# Load the complete dataset
for encoding in encodings_to_try:
    try:
        data = pd.read_csv('merged_movies.csv', encoding=encoding)
        break  # Stop trying encodings if successful
    except UnicodeDecodeError:
        continue  # Try the next encoding

print(data.head())
print(data.describe())

# TESTING THE DATASET FOR NORMALITY


# Creating subplots for both scatter plots with trendlines
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Budget vs Worldwide Gross Revenue with trendline
sns.regplot(x='Budget', y='Worldwide_Gross', data=data, ax=axes[0])
axes[0].set_title('Budget vs Worldwide Gross Revenue')
axes[0].set_xlabel('Budget (in million Dollars)')
axes[0].set_ylabel('Worldwide Gross Revenue')

# Budget vs Rating with trendline
sns.regplot(x='Budget', y='Rating', data=data, ax=axes[1])
axes[1].set_title('Budget vs Movie Rating')
axes[1].set_xlabel('Budget (in million Dollars)')
axes[1].set_ylabel('Rating')

plt.tight_layout()
plt.show()

# Testing for normality in the Budget, Worldwide Gross, and Rating columns
_, p_value_budget = stats.normaltest(data['Budget'])
_, p_value_worldwide_gross = stats.normaltest(data['Worldwide_Gross'])
_, p_value_rating = stats.normaltest(data['Rating'])

print(f'Normality test p-value for Budget: {p_value_budget}')
print(f'Normality test p-value for Worldwide Gross: {p_value_worldwide_gross}')
print(f'Normality test p-value for Rating: {p_value_rating}')

# Calculate Spearman correlation coefficient and p-value for Budget vs Worldwide Gross and Budget vs Rating
spearman_corr_gross, p_value_gross = spearmanr(data['Budget'], data['Worldwide_Gross'])
spearman_corr_rating, p_value_rating = spearmanr(data['Budget'], data['Rating'])

print(f"Spearman's correlation coefficient (Budget vs Worldwide Gross): {spearman_corr_gross}")
print(f"P-value (Budget vs Worldwide Gross): {p_value_gross}")

print(f"Spearman's correlation coefficient (Budget vs Rating): {spearman_corr_rating}")
print(f"P-value (Budget vs Rating): {p_value_rating}")


# TRAINING DATA
# Splitting the data for worldwide gross revenue prediction
X_wg = data[['Budget']]  # Feature (Budget)
y_wg = data['Worldwide_Gross']  # Target variable (Worldwide Gross Revenue)
X_train_wg, X_test_wg, y_train_wg, y_test_wg = train_test_split(X_wg, y_wg, test_size=0.2, random_state=42)
# Splitting the data for movie rating prediction
X_rating = data[['Budget']]  # Feature (Budget)
y_rating = data['Rating']  # Target variable (Rating)
X_train_rating, X_test_rating, y_train_rating, y_test_rating = train_test_split(X_rating, y_rating, test_size=0.2, random_state=42)

# MODELLING USING LINEAR REGRESSION

# Linear Regression model for worldwide gross revenue prediction
model_lr_wg = LinearRegression()
model_lr_wg.fit(X_train_wg, y_train_wg)
y_pred_lr_wg = model_lr_wg.predict(X_test_wg)

# Model evaluation for worldwide gross revenue prediction using Linear Regression
print("Worldwide Gross Revenue Prediction Model (Linear Regression):")
print(f'Coefficient: {model_lr_wg.coef_[0]}')
print(f'Intercept: {model_lr_wg.intercept_}')
print(f'Mean Squared Error: {mean_squared_error(y_test_wg, y_pred_lr_wg)}')
print(f'R-squared: {r2_score(y_test_wg, y_pred_lr_wg)}')

# Linear Regression model for movie rating prediction
model_lr_rating = LinearRegression()
model_lr_rating.fit(X_train_rating, y_train_rating)
y_pred_lr_rating = model_lr_rating.predict(X_test_rating)

# Model evaluation for movie rating prediction using Linear Regression
print("\nMovie Rating Prediction Model (Linear Regression):")
print(f'Coefficient: {model_lr_rating.coef_[0]}')
print(f'Intercept: {model_lr_rating.intercept_}')
print(f'Mean Squared Error: {mean_squared_error(y_test_rating, y_pred_lr_rating)}')
print(f'R-squared: {r2_score(y_test_rating, y_pred_lr_rating)}')



# MODELLING USING RANDOM FOREST

# Random Forest model for worldwide gross revenue prediction
model_rf_wg = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf_wg.fit(X_train_wg, y_train_wg)
y_pred_rf_wg = model_rf_wg.predict(X_test_wg)

# Model evaluation for worldwide gross revenue prediction using Random Forest
print("Worldwide Gross Revenue Prediction Model (Random Forest):")
print(f'Mean Squared Error: {mean_squared_error(y_test_wg, y_pred_rf_wg)}')
print(f'R-squared: {r2_score(y_test_wg, y_pred_rf_wg)}')



# Random Forest model for movie rating prediction
model_rf_rating = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf_rating.fit(X_train_rating, y_train_rating)
y_pred_rf_rating = model_rf_rating.predict(X_test_rating)

# Model evaluation for movie rating prediction using Random Forest
print("\nMovie Rating Prediction Model (Random Forest):")
print(f'Mean Squared Error: {mean_squared_error(y_test_rating, y_pred_rf_rating)}')
print(f'R-squared: {r2_score(y_test_rating, y_pred_rf_rating)}')