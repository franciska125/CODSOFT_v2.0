# Import necessary libraries
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

names = []
genres = []
descriptions = []

# Input File Path
file_path = r'C:\Users\BVM\PycharmProjects\CODSOFT_v1.0\Genre Classification\Genre Classification Dataset\train_data.txt'

with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # Split each line based on triple colons
        parts = line.strip().split(':::')
        # Ensure there are at least three parts (name, genre, description)
        if len(parts) >= 4:
            sno = parts[0].strip()  # Extracting the movie name
            name = parts[1].strip()  # Extracting the movie name
            genre = parts[2].strip()  # Extracting the genre
            description = parts[3].strip()  # Extracting the description

            # Appending list
            names.append(name)
            genres.append(genre)
            descriptions.append(description)

data = {'Plot': descriptions,
        'Genre': genres}
df = pd.DataFrame(data)


# Splitting dataset into training and test datasets
train_data, test_data, train_labels, test_labels = train_test_split(
    df['Plot'], df['Genre'], test_size=0.2, random_state=42
)

# Codes for TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5, max_df=0.8)
tfidf_train = tfidf_vectorizer.fit_transform(train_data)
tfidf_test = tfidf_vectorizer.transform(test_data)

# Initialize and train the Logistic Regression model
model = LogisticRegression(C=1.0, max_iter=1000)
model.fit(tfidf_train, train_labels)

# Prediction on test data
predictions = model.predict(tfidf_test)

# Model Evaluation
accuracy = accuracy_score(test_labels, predictions)
print(f"Accuracy: {accuracy:.2f}")

# Classification report and Confusion Matrix
print("\nClassification Report:")
print(classification_report(test_labels, predictions))

# Confusion Matrix
conf_matrix = pd.crosstab(test_labels, predictions, rownames=['Actual'], colnames=['Predicted'])
print("\nConfusion Matrix:")
print(conf_matrix)

# Trained model
with open('logistic_regression_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Plot Confusion Matrix using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=conf_matrix.columns, yticklabels=conf_matrix.index)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()