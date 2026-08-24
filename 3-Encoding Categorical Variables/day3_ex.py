import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load Titanic Dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Display dataset information
print("Dataset Info:")
print(df.info())

# Preview the first few rows
print("\n Dataset Preview:")
print(df.head())

# Applt One-Hot Encoding
df_one_hot = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Display encoded dataset
print("\n One-Hot Encoded dataset:")
print(df_one_hot.head())

# Apply Label Encoding
label_encoder = LabelEncoder()
df['Pclass_encoded'] = label_encoder.fit_transform(df['Pclass'])

# Display encoded dataset
print("\n Label Encoded Dataset:")
print(df[['Pclass', 'Pclass_encoded']].head())

# APply Frequency Encoding
df['Ticket_frequency'] = df['Ticket'].map(df['Ticket'].value_counts())

# Display frequency encoded feature
print("\n Frequence ENcoded Feature: ")
print(df[['Ticket', 'Ticket_frequency']].head())

X = df_one_hot.drop(columns=['Survived', 'Name', 'Ticket', 'Cabin', 'Age'])
y = df['Survived']


# SPlit dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy with One-Hot Encoding:", accuracy_score(y_test, y_pred))

