import pickle
import tkinter as tk
from tkinter import scrolledtext

# Loading trained model
with open('logistic_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Loading TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Prediction function
def predict_genre():
    input_text = input_text_widget.get("1.0", "end-1c")  # Get text from the input widget
    input_tfidf = tfidf_vectorizer.transform([input_text])  # Vectorize the input text
    prediction = model.predict(input_tfidf)  # Make prediction
    prediction_prob = model.predict_proba(input_tfidf)  # Get prediction probabilities
    accuracy = max(prediction_prob[0])  # Accuracy is the maximum probability
    result_label.config(text=f"Predicted Genre: {prediction[0]}")

# Application Development
root = tk.Tk()
root.title("Genre Prediction")

# Text input widget
input_text_widget = scrolledtext.ScrolledText(root, width=70, height=13, wrap=tk.WORD)
input_text_widget.grid(row=0, column=0, padx=10, pady=10)

# Predict button
predict_button = tk.Button(root, text="Predict Genre", command=predict_genre)
predict_button.grid(row=1, column=0, padx=10, pady=10)

# Result label
result_label = tk.Label(root, text="Predicted Genre: ")
result_label.grid(row=2, column=0, padx=10, pady=10)

# Run the GUI
root.mainloop()
