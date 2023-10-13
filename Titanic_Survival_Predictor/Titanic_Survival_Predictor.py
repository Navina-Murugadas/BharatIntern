import streamlit as st
import pandas as pd
import numpy as np
import pickle

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

st.title('TITANIC SURVIVAL PREDICTOR')

def predict(Pclass, Sex, Age, Sibsp, Parch, Fare, Embarked):
    sex_map = {'male': 0, 'female': 1}
    embarked_map = {'S': 0, 'C': 1, 'Q': 2}

    sex_encode = sex_map[Sex]  
    embarked_encode = embarked_map[Embarked]

    input_data = np.array([Pclass, sex_encode, Age, Sibsp, Parch, Fare, embarked_encode]).reshape(1, -1)
    prediction = model.predict(input_data)

    return prediction[0]
    
pclass = st.selectbox('Pclass (Passenger Class)', [1, 2, 3])
sex = st.radio('Sex', ['Male', 'Female'])
age = st.slider('Age', 0, 100, 30)
sibsp = st.slider('Siblings/Spouses Aboard', 0, 8, 0)
parch = st.slider('Parents/Children Aboard', 0, 6, 0)
fare = st.number_input('Fare')
embarked = st.radio('Embarked', ['S', 'C', 'Q'])

if st.button('PREDICT'):
    sex = "male" if sex == "Male" else "female"
    prediction = predict(pclass, sex, age, sibsp, parch, fare, embarked)
    if prediction == 1:
        st.header('SURVIVED')
    else:
        st.header('DEAD')
