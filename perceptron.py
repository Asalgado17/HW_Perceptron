import numpy as np
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)


path_data = 'perceptron.csv'

# Read the raw data
perceptron_raw_data = pd.read_csv(path_data, delimiter=",")
perceptron_raw_data.LT = perceptron_raw_data.LT.map({'T': 1, 'L': -1})
perceptron_raw_data_np = perceptron_raw_data.to_numpy()
np.random.shuffle(perceptron_raw_data_np)
X_test = perceptron_raw_data_np[60:97, :16]  # Features
y_test = perceptron_raw_data_np[60:97, 16]   # Labels

# Activation function for perceptron model
def activation_function(z):
    return np.where(z >= 0, 1, -1)

# Function to predict using perceptron model
def predict(X_test, weights, bias):
    linear_product = np.dot(X_test, weights) + bias  # y = w * x + b
    y_pred = activation_function(linear_product)
    return y_pred

# Route to handle requests and display the form
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None  # Default prediction result

    # Load pre-trained model weights and bias
    weights = np.load('weights.npy')
    bias = np.load('bias.npy')

    if request.method == 'POST':
        # List to store the input values from the form
        inputs = []

        # grabs  values from the dropdown box
        for row in range(4):
            for col in range(4):
                input_value = request.form.get(f'input_{row}_{col}')
                
                # Default to 0 if no value is selected since it gives issues with 'none'
                if input_value is None:
                    input_value = 0
                else:
                    input_value = int(input_value)
                
                inputs.append(input_value)  # Append input value to the empty list

        
        input_array = np.array(inputs).reshape(1, 16)

        # Make prediction using the perceptron model
        y_pred = predict(input_array, weights, bias)

        # Convert prediction to 'T' or 'L'
        prediction_result = 'T' if y_pred == 1 else 'L'

    
    return render_template('index.html', prediction=prediction_result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)
