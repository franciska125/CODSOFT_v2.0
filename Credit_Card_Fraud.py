import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt

# Load the training dataset
df_train = pd.read_csv('../Credit_Card_Fraud_Detection/Dataset/fraudTrain.csv')  # Replace with the actual training dataset filename

# Feature selection
features = ['amt', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long']
X_train = df_train[features]
y_train = df_train['is_fraud']

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Build and train the Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Save the trained model
joblib.dump(model, 'fraud_detection_model.joblib')

# Load the test dataset
df_test = pd.read_csv('../Credit_Card_Fraud_Detection/Dataset/fraudTest.csv')  # Replace with the actual test dataset filename

# Preprocess the test dataset
X_test = df_test[features]
X_test_scaled = scaler.transform(X_test)
y_test = df_test['is_fraud']

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model on the test set
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the test set: {accuracy:.2%}")

# Print classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Print cases where fraud is predicted
fraudulent_cases = df_test[y_pred == 1]

# Print cases where legitimate is predicted
legitimate_cases = df_test[y_pred == 0]

# Save predicted fraudulent and legitimate cases to separate Excel files
fraudulent_cases.to_csv('predicted_fraudulent_cases.csv', index=False)
legitimate_cases.to_csv('predicted_legitimate_cases.csv', index=False)

# Plot confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Legitimate', 'Fraudulent'], yticklabels=['Legitimate', 'Fraudulent'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
