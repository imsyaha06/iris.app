import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import joblib
from PIL import Image

# Load pre-trained ANN model and scaler
model = load_model ('iris_ann_model.keras')
scaler = joblib.load('iris_scaler.pkl')

# Load species images
setosa = Image.open('setosa.png')
versicolor = Image.open('versicolor.png')
virginica = Image.open('virginica.png')

# Title of the Streamlit app
st.title("Iris Species Prediction using ANN")

# Sidebar for user input
st.sidebar.header("Input Features")

# Initialize input parameters and default values
parameter_list = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']  # Match with model's training names
default_values = [5.2, 3.2, 4.2, 1.2]

# Create a dictionary to store the input values
parameter_input_values = {}

# Create sliders for user input
for parameter, default_value in zip(parameter_list, default_values):
    parameter_input_values[parameter] = st.sidebar.slider(label=parameter.replace('Cm', ' (cm)'), value=default_value, min_value=0.0, max_value=10.0, step=0.1)

# Convert the input values to a DataFrame
input_variables = pd.DataFrame([parameter_input_values], columns=parameter_list, dtype=float)

# # Display the input features in the main page
# st.subheader("Input Features")
# st.write(input_variables)

# Preprocess the user input using the saved scaler
input_scaled = scaler.transform(input_variables)

# Make predictions
prediction = np.argmax(model.predict(input_scaled), axis=1)

# Map prediction back to species names
species_dict = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
prediction_species = species_dict[prediction[0]]

# Display the prediction
st.subheader("Prediction")
st.write(f"The predicted species is: **{prediction_species}**")

# Display corresponding species image
if prediction[0] == 0:
    st.image(setosa, caption="Setosa")
elif prediction[0] == 1:
    st.image(versicolor, caption="Versicolor")
else:
    st.image(virginica, caption="Virginica")
