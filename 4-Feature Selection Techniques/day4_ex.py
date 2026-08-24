from sklearn.datasets import load_diabetes
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression

# Load the dataset
data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Display Dataset information
# print(df.head())
# print(df.info())

# Calculate correlation matrix
correlation_matrix = df.corr()

# Plot heatmap
# plt.figure(figsize=(10,8))
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
# plt.title("Correlation Matrix")
# plt.show()

# Select features with high correlation to the target
correlated_features = correlation_matrix['target'].sort_values(ascending=False)
# print("Features Most Correlated with Target:")
# print(correlated_features)

# Seperate featured and target
X = df.drop(columns=['target'])
y = df['target']

# Calculate mutual information
mutual_info = mutual_info_regression(X, y)

# Create a Dataframe for better visualization
mi_df = pd.DataFrame({'Feature': X.columns, "Mutual Information": mutual_info})
mi_df = mi_df.sort_values(by="Mutual Information", ascending=False)

# print("Mutual Information Scores:")
# print(mi_df)

from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Train a Random Forest Model
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Get feature importance
feature_importance = model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("Feature Importance from Random Forest:")
print(importance_df)

# PLot feature importance
plt.figure(figsize=(10,6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.gca().invert_yaxis()
plt.title("Feature Importance from Random Forest")
plt.show()