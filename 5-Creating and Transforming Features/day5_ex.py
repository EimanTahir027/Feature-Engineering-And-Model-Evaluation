import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load Bike Sharing Dataset
df = pd.read_csv("bike_sharing_daily.csv")

# Display dataset information 
# print("Dataset Info:")
# print(df.info())

# Preview the first few rows
# print("\n Dataset Preview:")
# print(df.head())

# Convert dteday to datetime
df['dteday'] = pd.to_datetime(df['dteday'])

# Create new features
df['day_of_week'] = df['dteday'].dt.day_name()
df['month'] = df['dteday'].dt.month
df['year'] = df['dteday'].dt.year

# Display the enw features
# print("\n New Features Derived from Date Column")
# print(df[['dteday', 'day_of_week', 'month', 'year']].head())

# Select feature and target
X = df[['temp']]
y = df['cnt']

# Apply polynomial transformation
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Display the tranformed feature
# print("\n Original and Polynomial Features")
# print(pd.DataFrame(X_poly, columns=['temp', 'temp^2']).head())

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_poly_train, X_poly_test = train_test_split(X_poly, test_size=0.2, random_state=42)

# Train and evaluate model with original features
model_original = LinearRegression()
model_original.fit(X_train, y_train)
y_pred_original = model_original.predict(X_test)
mse_original = mean_squared_error(y_test, y_pred_original)

# Train and evaluate model with polynomial features
model_poly = LinearRegression()
model_poly.fit(X_poly_train, y_train)
y_pred_poly = model_poly.predict(X_poly_test)
mse_poly = mean_squared_error(y_test, y_pred_poly)

# COmpare results
print(f"MSE original: {mse_original:.2f}")
print(f"MSE Polynomial: {mse_poly:.2f}")