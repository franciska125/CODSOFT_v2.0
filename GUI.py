import joblib
import tkinter as tk
from tkinter import scrolledtext
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model
model = joblib.load(r'C:\Users\Franciska Fdo\PycharmProjects\CODSOFT_v2.0_new\SMS_Classifier\spam_classifier_model.pkl')

# Load the TF-IDF vectorizer
tfidf_vectorizer = joblib.load(r'C:\Users\Franciska Fdo\PycharmProjects\CODSOFT_v2.0_new\SMS_Classifier\tfidf_vectorizer.pkl')

# Ensure tfidf_vectorizer is an instance of TfidfVectorizer
if not isinstance(tfidf_vectorizer, TfidfVectorizer):
    raise TypeError("The loaded tfidf_vectorizer is not an instance of TfidfVectorizer.")

# Prediction function
def predict_spam():
    try:
        input_text = input_text_widget.get("1.0", "end-1c")  # Get text from the input widget
        input_tfidf = tfidf_vectorizer.transform([input_text])  # Vectorize the input text
        prediction = model.predict(input_tfidf)  # Make prediction
        print('prediction',prediction)
        result_label.config(text=f"Prediction: {'Spam' if prediction[0] == 'spam' else 'Legitimate'}")

    except Exception as e:
        result_label.config(text=f"Error: {e}")

# Application Development
root = tk.Tk()
root.title("SMS Spam Detection")

# Text input widget
input_text_widget = scrolledtext.ScrolledText(root, width=70, height=13, wrap=tk.WORD)
input_text_widget.grid(row=0, column=0, padx=10, pady=10)

# Predict button
predict_button = tk.Button(root, text="Predict Spam", command=predict_spam)
predict_button.grid(row=1, column=0, padx=10, pady=10)

# Result label
result_label = tk.Label(root, text="Prediction: ")
result_label.grid(row=2, column=0, padx=10, pady=10)

# Run the GUI
root.mainloop()
