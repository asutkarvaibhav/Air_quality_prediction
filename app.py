from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
model_path = 'model.pkl'  
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input features from the form
        features = ['Temperature', 'NO2', 'SO2', 'Proximity_to_Industrial_Areas', 'CO']
        int_features = [float(request.form[feature]) for feature in features]
        final_features = [np.array(int_features)]
        
        # Make prediction
        prediction = model.predict(final_features)
        output_map = {0: 'Good', 1: 'Hazardous', 2: 'Moderate', 3: 'Poor'}
        output = output_map.get(prediction[0], 'Unknown')
        
        return render_template('index.html', prediction_text=f'Air Quality: {output}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
