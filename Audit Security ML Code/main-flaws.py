import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
# Load dataset (Flaw: No data validation or sanitization)
data = pd.read_csv('user_data.csv')
# Split the dataset into features and target (Flaw: No input validation)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
# Split the data into training and testing sets (Flaw: Fixed random state)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train a simple logistic regression model (Flaw: No model security checks)
model = LogisticRegression()
model.fit(X_train, y_train)
# Save the model to disk (Flaw: Unencrypted model saving)
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
# Load the model from disk for later use (Flaw: No integrity checks on the loaded model)
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(f'Model Accuracy: {result:.2f}')