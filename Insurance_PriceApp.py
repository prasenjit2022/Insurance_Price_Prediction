import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle

from PIL import Image
st.title("Insurance Price Prediction")
st.write("This app will predict the price of health insurance.")
image=Image.open('HealthInsurance.jpg')
st.image(image)
Name=st.text_input('Name')
Age=st.text_input('Age')
Bmi=st.text_input('BMI')
Region=st.selectbox('Region',('Southwest','Southeast','Northwest','Northeast'))
Sex=st.selectbox('Sex',('Male','Female'))
smoker=st.selectbox('Do you smoke?',('Yes','No'))
n_children=st.text_input('Number of Children')
user_details={'age':Age,'bmi':Bmi,'sex':Sex,'smoker':smoker,'children':n_children,'region':Region}
df=pd.DataFrame(user_details,index=[0])
df['sex']=df['sex'].map({'Female':1,'Male':0})
df['smoker']=df['smoker'].map({'Yes':1,'No':0})
df['region']=df['region'].map({'Northwest':1,'Southwest':2,'Northeast':3,'Southeast':4})
if st.button('Submit'):
    with open('final_model_gb.pkl','rb') as r:
        model=pickle.load(r)
        predict=model.predict(df)
    st.write(f"Price of Health Insurance is =${predict}")
