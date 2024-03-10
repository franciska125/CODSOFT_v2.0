import tkinter as tk
from tkinter import ttk
import joblib
import numpy as np

# Load the saved model
model = joblib.load('fraud_detection_model.joblib')

# Function to classify transaction
def classify_transaction():
    try:
        # Preprocess user input
        user_input = [float(entry_amt.get()), float(entry_lat.get()), float(entry_long.get()),
                      float(entry_city_pop.get()), float(entry_unix_time.get()),
                      float(entry_merch_lat.get()), float(entry_merch_long.get())]

        # Standardize the user input
        user_input_scaled = np.array(user_input).reshape(1, -1)

        # Make a prediction
        prediction = model.predict(user_input_scaled)
        print('prediction',prediction[0])
        # Display the prediction
        result_label.config(text=f"Prediction: {'Fraudulent' if prediction[0] == 1 else 'Legitimate'}")

    except ValueError:
        result_label.config(text="Invalid input. Please enter numerical values.")

# Create GUI window
root = tk.Tk()
root.title("Credit Card Fraud Detection")

# Entry fields
entry_amt = ttk.Entry(root, width=20)
entry_lat = ttk.Entry(root, width=20)
entry_long = ttk.Entry(root, width=20)
entry_city_pop = ttk.Entry(root, width=20)
entry_unix_time = ttk.Entry(root, width=20)
entry_merch_lat = ttk.Entry(root, width=20)
entry_merch_long = ttk.Entry(root, width=20)

# Labels
label_amt = ttk.Label(root, text="Transaction Amount:")
label_lat = ttk.Label(root, text="Latitude:")
label_long = ttk.Label(root, text="Longitude:")
label_city_pop = ttk.Label(root, text="City Population:")
label_unix_time = ttk.Label(root, text="Unix Time:")
label_merch_lat = ttk.Label(root, text="Merchant Latitude:")
label_merch_long = ttk.Label(root, text="Merchant Longitude:")

# Result label
result_label = ttk.Label(root, text="Prediction: ")

# Button
classify_button = ttk.Button(root, text="Classify", command=classify_transaction)

# Grid layout
label_amt.grid(row=0, column=0, padx=5, pady=5)
entry_amt.grid(row=0, column=1, padx=5, pady=5)
label_lat.grid(row=1, column=0, padx=5, pady=5)
entry_lat.grid(row=1, column=1, padx=5, pady=5)
label_long.grid(row=2, column=0, padx=5, pady=5)
entry_long.grid(row=2, column=1, padx=5, pady=5)
label_city_pop.grid(row=3, column=0, padx=5, pady=5)
entry_city_pop.grid(row=3, column=1, padx=5, pady=5)
label_unix_time.grid(row=4, column=0, padx=5, pady=5)
entry_unix_time.grid(row=4, column=1, padx=5, pady=5)
label_merch_lat.grid(row=5, column=0, padx=5, pady=5)
entry_merch_lat.grid(row=5, column=1, padx=5, pady=5)
label_merch_long.grid(row=6, column=0, padx=5, pady=5)
entry_merch_long.grid(row=6, column=1, padx=5, pady=5)

classify_button.grid(row=8, column=0, columnspan=2, pady=10)
result_label.grid(row=9, column=0, columnspan=2)

root.mainloop()
