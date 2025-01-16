"""
This script performs data cleaning and preprocessing on a given dataset.
It includes handling missing values, removing outliers, scaling the data, 
and encoding categorical variables.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler # MinMaxScaler
# import missingno as msno
from scipy import stats

# def load_data(file_path):
#     """
#     Load the data from the given file path.
#     """
#     return pd.read_csv(file_path)

def load_data(df):
    """
    Load the data from the given file path.
    """
    return df

def handle_missing_values(df):
    """
    Fill missing values:
    - Numeric columns: filled with column mean
    - Categorical columns: filled with mode (most frequent value)
    """
    # Handle numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].mean())
    # Handle categorical columns
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns
    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df

def remove_outliers(df):
    """
    Remove outliers using z-score.
    """
    z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
    return df[(z_scores < 3).all(axis=1)]

def scale_data(df):
    """
    Scale the data using StandardScaler.
    """
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def encode_categorical_data(df, categorical_columns):
    """
    Encode categorical data using one-hot encoding.
    """
    return pd.get_dummies(df, columns=categorical_columns)

def save_data(df, output_filepath):
    """
    Save the cleaned data to a new CSV file.
    """
    df.to_csv(output_filepath, index=False)

# Create a dummy dataset
np.random.seed(0)
dummy_data = {
    'Feature1': np.random.normal(100, 10, 100).tolist() + 
                [np.nan, 200],  # Normally distributed with an outlier
    'Feature2': np.random.randint(0, 100, 102).tolist(),  # Random integers
    'Category': ['A', 'B', 'C', 'D'] * 25 + [np.nan, 'A'],  # Categorical with some missing values
    'Target': np.random.choice([0, 1], 102).tolist()  # Binary target variable
}

# Convert the dictionary to a pandas DataFrame
df_dummy = pd.DataFrame(dummy_data)

# Display the first few rows of the dummy dataset
#print(df_dummy.head())

def main():
    """
    Main function to execute the data cleaning and preprocessing steps.
    """
    # input_filepath = 'cloud_computing_comparison.csv'
    output_filepath = 'cleaned_data.csv'
    categorical_columns = ['Category']

    df = load_data(df_dummy)
    df = handle_missing_values(df)
    df = remove_outliers(df)
    df = scale_data(df)
    df = encode_categorical_data(df, categorical_columns)
    print(df.head())
    save_data(df, output_filepath)

    # Check for missing values
    print(df.isnull().sum())

    # Verify outlier removal
    print(df.describe())

    # Inspect scaled data
    print(df.head())

    # Check categorical encoding
    print(df.columns)

if __name__ == "__main__":
    main()
