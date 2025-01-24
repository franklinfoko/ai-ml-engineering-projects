import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import missingno as msno
from scipy import stats

# Load the dataset into a pandas dataframe
df = pd.read_csv('cloud_computing_comparison.csv')

# Display the first few rows of the dataframe
print(df.head())

# Visualize the missing data
msno.matrix(df)
msno.heatmap(df)

# Drop rows with missing values
df_cleaned = df.dropna()

# Or, fill missing values with the mean of the column
df_filled = df.fillna(df.mean())

# Identify and remove outliers
z_scores = np.abs(stats.zscore(df_cleaned))
df_no_outliers = df_cleaned[(z_scores < 3).all(axis=1)]

# Or cap outliers at a threshold
upper_limit = df_cleaned['column_name'].quantile(0.95)
df_cleaned['column_name'] = np.where(df_cleaned['column_name'] > upper_limit, upper_limit, df_cleaned['column_name'])

# Min-max scaling
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_cleaned), columns=df_cleaned.columns)

# Z-score Standardization
scaler = StandardScaler()
df_standardized = pd.DataFrame(scaler.fit_transform(df_cleaned), columns=df_cleaned.columns)

# One-hot encoding for categorical variables
df_encoded = pd.get_dummies(df_scaled, columns=['categorical_column_name'])

# Save the cleaned dataframe to a new CSV file
df_encoded.to_csv('cleaned_data.csv', index=False)

print('Data cleaning and preprocessing complete. File saved as cleaned_preprocessed_data.csv')