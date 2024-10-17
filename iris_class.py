# import streamlit as st
# import numpy as np
# import joblib
# # import tensorflow as tf
# from tensorflow.keras.models import load_model

# # Load the saved ANN model and scaler
# model = load_model('iris_ann_model.keras')
# scaler = joblib.load('iris_scaler.pkl')
# feature_names = joblib.load('iris_feature_names.pkl')

# # Species names to match model output
# species_names = ['Setosa', 'Versicolor', 'Virginica']

# # Streamlit app title
# st.title("Iris Species Prediction using ANN")

# # Input fields for sepal and petal dimensions
# sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, max_value=10.0, value=5.1)
# sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, max_value=10.0, value=3.5)
# petal_length = st.number_input('Petal Length (cm)', min_value=0.0, max_value=10.0, value=1.4)
# petal_width = st.number_input('Petal Width (cm)', min_value=0.0, max_value=10.0, value=0.2)

# # Button to trigger prediction
# if st.button('Predict Species'):
#     # Prepare input data for prediction
#     input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
#     # Scale the input data using the saved scaler
#     input_data_scaled = scaler.transform(input_data)
    
#     # Make prediction using the loaded model
#     prediction = model.predict(input_data_scaled)
#     predicted_class = np.argmax(prediction, axis=1)[0]
    
#     # Display the prediction result
#     st.success(f"The predicted species is: **{species_names[predicted_class]}**")

# # Optionally, display the entered values
# st.subheader("Entered values:")
# st.write(f"Sepal Length: {sepal_length} cm")
# st.write(f"Sepal Width: {sepal_width} cm")
# st.write(f"Petal Length: {petal_length} cm")
# st.write(f"Petal Width: {petal_width} cm")














# import streamlit as st
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from tensorflow.keras.models import load_model
# from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report
# import joblib

# # Load pre-trained model, scaler, and feature names
# model = load_model('iris_ann_model.keras')
# scaler = joblib.load('iris_scaler.pkl')
# feature_names = joblib.load('iris_feature_names.pkl')

# # Title of the Streamlit app
# st.title("Iris Species Prediction using ANN")

# # Sidebar for user input
# st.sidebar.header("Input Features")
# def user_input_features():
#     SepalLengthCm = st.sidebar.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0)
#     SepalWidthCm = st.sidebar.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0)
#     PetalLengthCm = st.sidebar.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.5)
#     PetalWidthCm = st.sidebar.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)
    
#     data = {'SepalLengthCm': SepalLengthCm,
#             'SepalWidthCm': SepalWidthCm,
#             'PetalLengthCm': PetalLengthCm,
#             'PetalWidthCm': PetalWidthCm}
#     features = pd.DataFrame(data, index=[0])
#     return features

# # Collect user input features
# input_df = user_input_features()

# # Display the input features in the main page
# st.subheader("Input Features")
# st.write(input_df)

# # Preprocess the user input using the saved scaler
# input_scaled = scaler.transform(input_df)

# # Make predictions
# prediction = np.argmax(model.predict(input_scaled), axis=1)

# # Map prediction back to species names
# species_dict = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
# prediction_species = species_dict[prediction[0]]

# # Display the prediction
# st.subheader("Prediction")
# st.write(f"The predicted species is: **{prediction_species}**")

# # Option to show model performance (this part can show classification report for test data)
# if st.checkbox("Show Model Performance on Test Data"):
#     # Load the Iris dataset again for testing
#     df = pd.read_csv("Iris.csv")
#     df = df.drop("Id", axis=1)

#     # Preprocessing (same as earlier)
#     X = df.drop("Species", axis=1)
#     y = df["Species"]
#     label_encoder = LabelEncoder()
#     y = label_encoder.fit_transform(y)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Normalize the data
#     X_test = scaler.transform(X_test)

#     # Predictions on test data
#     y_pred = np.argmax(model.predict(X_test), axis=1)

#     # Calculate accuracy, precision, F1 score
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred, average='weighted')
#     f1 = f1_score(y_test, y_pred, average='weighted')
    
#     # Classification report
#     report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    
#     # Display metrics
#     st.subheader("Model Performance")
#     st.write(f"Test Accuracy: {accuracy * 100:.2f}%")
#     st.write(f"Test Precision: {precision * 100:.2f}%")
#     st.write(f"Test F1 Score: {f1 * 100:.2f}%")
#     st.text("\nClassification Report:\n")
#     st.text(report)

# # Footer
# st.write("Created with ❤️ by Sumbal")










# import streamlit as st
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from tensorflow.keras.models import load_model
# import joblib
# from PIL import Image

# # Load pre-trained ANN model and scaler
# model = load_model('iris_ann_model.keras')
# scaler = joblib.load('iris_scaler.pkl')

# # Load species images
# setosa = Image.open('setosa.png')
# versicolor = Image.open('versicolor.png')
# virginica = Image.open('virginica.png')

# # Title of the Streamlit app
# st.title("Iris Species Prediction using ANN")

# # Sidebar for user input
# st.sidebar.header("Input Features")

# def user_input_features():
#     SepalLengthCm = st.sidebar.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0)
#     SepalWidthCm = st.sidebar.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0)
#     PetalLengthCm = st.sidebar.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.5)
#     PetalWidthCm = st.sidebar.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)
    
#     data = {
#         'SepalLengthCm': SepalLengthCm,
#         'SepalWidthCm': SepalWidthCm,
#         'PetalLengthCm': PetalLengthCm,
#         'PetalWidthCm': PetalWidthCm
#     }
#     features = pd.DataFrame(data, index=[0])
#     return features

# # Collect user input features
# input_df = user_input_features()

# # # Display the input features in the main page
# # st.subheader("Input Features")
# # st.write(input_df)

# # Preprocess the user input using the saved scaler
# input_scaled = scaler.transform(input_df)

# # Make predictions
# prediction = np.argmax(model.predict(input_scaled), axis=1)

# # Map prediction back to species names
# species_dict = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
# prediction_species = species_dict[prediction[0]]

# # Display the prediction
# st.subheader("Prediction")
# st.write(f"The predicted species is: **{prediction_species}**")

# # Display corresponding species image
# if prediction[0] == 0:
#     st.image(setosa, caption="Setosa")
# elif prediction[0] == 1:
#     st.image(versicolor, caption="Versicolor")
# else:
#     st.image(virginica, caption="Virginica")

# # # Option to show model performance (this part can show classification report for test data)
# # if st.checkbox("Show Model Performance on Test Data"):
# #     # Load the Iris dataset again for testing
# #     df = pd.read_csv("Iris.csv")
# #     df = df.drop("Id", axis=1)

# #     # Preprocessing (same as earlier)
# #     X = df.drop("Species", axis=1)
# #     y = df["Species"]
# #     label_encoder = LabelEncoder()
# #     y = label_encoder.fit_transform(y)

# #     # Normalize the data
# #     X_scaled = scaler.transform(X)

#     # # Predictions on test data
#     # y_pred = np.argmax(model.predict(X_scaled), axis=1)

#     # # Calculate accuracy, precision, F1 score
#     # accuracy = accuracy_score(y, y_pred)
#     # precision = precision_score(y, y_pred, average='weighted')
#     # f1 = f1_score(y, y_pred, average='weighted')
    
#     # # Classification report
#     # report = classification_report(y, y_pred, target_names=label_encoder.classes_)
    
# #     # Display metrics
# #     st.subheader("Model Performance")
# #     st.write(f"Test Accuracy: {accuracy * 100:.2f}%")
# #     st.write(f"Test Precision: {precision * 100:.2f}%")
# #     st.write(f"Test F1 Score: {f1 * 100:.2f}%")
# #     st.text("\nClassification Report:\n")
# #     st.text(report)

# # # Footer
# # st.write("Created with ❤️ by Sumbal")










###########################################################################################################




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

# # Option to show model performance (this part can show classification report for test data)
# if st.checkbox("Show Model Performance on Test Data"):
#     # Load the Iris dataset again for testing
#     df = pd.read_csv("Iris.csv")
#     df = df.drop("Id", axis=1)

#     # Preprocessing (same as earlier)
#     X = df.drop("Species", axis=1)
#     y = df["Species"]
#     label_encoder = LabelEncoder()
#     y = label_encoder.fit_transform(y)

#     # Normalize the data
#     X_scaled = scaler.transform(X)

#     # Predictions on test data
#     y_pred = np.argmax(model.predict(X_scaled), axis=1)

#     # Calculate accuracy, precision, F1 score
#     accuracy = accuracy_score(y, y_pred)
#     precision = precision_score(y, y_pred, average='weighted')
#     f1 = f1_score(y, y_pred, average='weighted')
    
#     # Classification report
#     report = classification_report(y, y_pred, target_names=label_encoder.classes_)
    
#     # Display metrics
#     st.subheader("Model Performance")
#     st.write(f"Test Accuracy: {accuracy * 100:.2f}%")
#     st.write(f"Test Precision: {precision * 100:.2f}%")
#     st.write(f"Test F1 Score: {f1 * 100:.2f}%")
#     st.text("\nClassification Report:\n")
#     st.text(report)

# # Footer
# st.write("Created with ❤️ by Sumbal")



 