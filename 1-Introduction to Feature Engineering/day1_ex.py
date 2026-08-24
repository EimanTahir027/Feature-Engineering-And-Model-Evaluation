import pandas as pd

# Load Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Display dataset Information
# print("Dataset Info: \n")
# print(df.info())

# # Preview the first few rows
# print("\n Dataset Preview:\n")
# print(df.head())

# Seperate features
categorical_features = df.select_dtypes(include=["object"]).columns
numberical_features = df.select_dtypes(include=["int64", "float64"]).columns

print("\nCategorical Features: ", categorical_features.tolist())
print("\nNumerical Features: ", numberical_features.tolist())

# Display summary of categorical features
print("\n Categorical Feature Summary:\n")
for col in categorical_features:
    print(f"{col}:\n", df[col].value_counts(), "\n")
    
# Display Summary of numerical features
print("\n Numerical Feature Summary:\n")
print(df[numberical_features].describe())
