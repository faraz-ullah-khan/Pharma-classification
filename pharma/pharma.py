import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
# with open('std.pkl', 'rb') as std_file:
#     std = pickle.load(std_file)
# App title
st.title("Liver Disease Prediction App")
st.write("Predict the likelihood of liver disease based on patient information.")

# Sidebar inputs
st.sidebar.header("Input Patient Details")

def user_input_features():
    

    Age= st.text_input ('Age')
    Sex = st.text_input ('Sex')
    Albumin = st.text_input ('Albumin')
    Alkaline_phosphatase = st.text_input ('Alkaline_phosphatase')   
    Alanine_aminotransferase = st.text_input ('Alanine_aminotransferase')
    Aspartate_aminotransferase = st.text_input ('Aspartate_aminotransferase')
    Bilirubin = st.text_input ('Bilirubin')
    Cholinesterase = st.text_input ('Cholinesterase')
    Cholesterol = st.text_input ('Cholesterol')
    Creatinina = st.text_input ('Creatinina')
    Gamma_glutamyl_transferase = st.text_input ('Gamma_glutamyl_transferase')
    protein = st.text_input ('protein')
    
    Sex_numeric = 1 if Sex == "Male" else 0  # Convert categorical to numeric if needed

    data = {
        'Age': Age,
        'Sex': Sex,
        'Albumin': Albumin,
        'Alkaline_phosphatase': Alkaline_phosphatase,
        'Alanine_aminotransferase': Alanine_aminotransferase,
        'Aspartate_aminotransferase': Aspartate_aminotransferase,
        'Bilirubin': Bilirubin,
        'Cholinesterase': Cholinesterase,
        'Cholesterol': Cholesterol,
        'Creatinina': Creatinina,
        'Gamma_glutamyl_transferase': Gamma_glutamyl_transferase,
        'protein': protein,
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display user input
st.subheader("Patient Details")
st.write(input_df)


category_mapping = {
    3: 'no disease',
    4: 'suspect_disease',
    2: 'hepatitis',
    1: 'fibrosis',
    0: 'cirrhosis'
}


scaler = StandardScaler()
# Create a 'Predict' button
if st.button('Predict'):
    # Prediction
    std_num = scaler.fit_transform(input_df)
    # std_num=input_df.transform(input_df)
    prediction = model.predict(std_num)
    prediction_proba = model.predict_proba(std_num)

    # Display the prediction result
    predicted_category_name = category_mapping.get(prediction[0], 'Unknown')
    st.subheader('Predicted Result')
    st.write(predicted_category_name)

    # Display the prediction probability
    st.subheader('Prediction Probability')
    st.write(prediction_proba)



     