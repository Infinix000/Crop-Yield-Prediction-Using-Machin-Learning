# app.py

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv("yield_df.csv")

# Load the trained model
model_path = 'Lightbgm.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)
    
area_encoder = LabelEncoder()
item_encoder = LabelEncoder()

area_encoder.fit(df['Area'])
item_encoder.fit(df['Item'])

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    area = request.form['Area']
    item = request.form['Item']
    rainfall = float(request.form['Rainfall'])
    pesticides = float(request.form['Pesticides'])
    avg_temp = float(request.form['Avg_Temp'])
    
    # Encode categorical variables
    area_encoded = area_encoder.transform([area])[0]  # Encode 'area'
    item_encoded = item_encoder.transform([item])[0]  # Encode 'item'

    # Prepare the data for prediction using encoded values
    input_data = np.array([[area_encoded, item_encoded, rainfall, pesticides, avg_temp]])
    #input_data = pd.DataFrame([[area_encoded, item_encoded, rainfall, pesticides, avg_temp]], 
                               #columns=['Country_Encoded', 'Crop_Encoded', 'Rainfall', 'Pesticides', 'Avg_Temp'])


    # Make prediction
    prediction = model.predict(input_data)

    # Render the result on the same page
    return render_template('index.html', prediction_text=f'Predicted Crop Yield: {prediction[0]}')

if __name__ == "__main__":
    app.run(debug=True)