from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd

# Load Iris dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Display dataset Information
print("Dataset Info:")
print(X.describe())
print("\n Target Classes:", data.target_names)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train k-NN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict and evaluate
y_pred = knn.predict(X_test)
print("Accuracy Without Scaling:", accuracy_score(y_test, y_pred))

# Apply Min-Max Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# SPlit scaled data
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train k-NN classifier on scaled data
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train_scaled)

# Predict and evaluate
y_pred_scaled = knn_scaled.predict(X_test_scaled)
print("Accuracy with Min-Max Scaling:", accuracy_score(y_test_scaled, y_pred_scaled))

# Apply Standardization
scaler = StandardScaler()
X_stand = scaler.fit_transform(X)

# SPlit standardized data
X_train_std, X_test_std, y_train_std, y_test_std = train_test_split(X_stand, y, test_size=0.2, random_state=42)

# Train k-NN classifier on standardized data
knn_stand = KNeighborsClassifier(n_neighbors=5)
knn_stand.fit(X_train_std, y_train_std)

# Predict and evaluate
y_pred_std = knn_stand.predict(X_test_std)
print("Accuracy with Standardization:", accuracy_score(y_test_std, y_pred_std))





