import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Loading Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Select relevant features
df = df[['Pclass','Sex','Age','Fare','Embarked','Survived']]

# Handle missing values
#df.method({col: value}, inplace=True)
df.fillna({'Age':df['Age'].median()}, inplace=True)
df.fillna({'Embarked':df['Embarked'].mode()[0]}, inplace=True)

# Define features and target
X = df.drop(columns=['Survived'])
y = df['Survived']

# Apply feature scaling and encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Age', 'Fare']),
        ('cat', OneHotEncoder(), ['Pclass', 'Sex', 'Embarked'])
    ]
)

X_preprocessed = preprocessor.fit_transform(X)

# Train and evaluate Logistic Regression
log_model = LogisticRegression()
log_scores = cross_val_score(log_model, X_preprocessed, y, cv=5, scoring='accuracy')
print(f"Logistic Regression Accuracy: {log_scores.mean():.2f}")

# Train and evaluate Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_scores = cross_val_score(rf_model, X_preprocessed, y, cv=5, scoring='accuracy')
print(f"Random Forest Accuracy: {rf_scores.mean():.2f}")

# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Perform Grid Search
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1
)
grid_search.fit(X_preprocessed, y)

# Display best hyperparameters and score
print(f"Best hyperparameters: {grid_search.best_params_}")
print(f"Best Accuracy: {grid_search.best_score_:.2f}")
