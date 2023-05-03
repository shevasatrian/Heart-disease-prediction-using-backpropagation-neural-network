import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
# import h5

st.title('Heart Disease Prediction Using BackPropagation Neural Network')

model = load_model('BPNN.h5')

firstCol  = st.container()
secondCol = st.container()
thirdCol = st.container()


with firstCol:
    col1, col2,col3,col4,col5 = firstCol.columns(5)
    
    with col1:
        age = st.number_input("Age", min_value=0, max_value=200, value=5, step=1, format="%d")
        # st.write('0', number)

    with col2:
        optionSex = st.selectbox(
            "Sex",
            ("M", "F")
        )

    with col3:
        optionCP = st.number_input("CP", min_value=0, max_value=3, value=0, step=1, format="%d")

    with col4:
        optionTrestbps = st.number_input("Trestbps", min_value=0, max_value=300, value=0, step=1, format="%d")

    with col5:
        optionChol = st.number_input("Chol", min_value=0, max_value=400, value=0, step=1, format="%d")

with secondCol:
    col6,col7,col8,col9,col10, = secondCol.columns(5)

    with col6:
        optionFbs = st.number_input("Fbs", min_value=0, max_value=1, value=0, step=1, format="%d")

    with col7:
        optionRestecg = st.number_input("Restecg", min_value=0, max_value=2, value=0, step=1, format="%d")

    with col8:
        optionThalach = st.number_input("Thalach", min_value=0, max_value=200, value=0, step=1, format="%d")

    with col9:
        optionExang = st.number_input("Exang", min_value=0, max_value=1, value=0, step=1, format="%d")

    with col10:
        optionOldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=5.0)

with thirdCol:
    col11,col12,col13 = thirdCol.columns(3)

    with col11:
        optionSlope = st.number_input("Slope", min_value=0, max_value=2, value=0, step=1, format="%d")
    
    with col12:
        optionCA = st.number_input("CA", min_value=0, max_value=4, value=0, step=1, format="%d")

    with col13:
        optionThal = st.number_input("Thal", min_value=0, max_value=3, value=0, step=1, format="%d")


click = st.button("Prediksi")

def pred():
    table = [age, optionSex, optionCP, optionTrestbps, optionChol, optionFbs, optionRestecg,
        optionThalach, optionExang, optionOldpeak, optionSlope, optionCA, optionThal]

    if(table[1]=="M"):
        table[1] = 0
    else:
        table[1] = 1

    untukPrediksi = np.array(table)
    input_data_reshaped =untukPrediksi.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)[0]

    pred_labels = []
    if prediction > 0.5:
        s = 'Heart Disease'
        pred_labels.append(0)
    else:
        s = 'Healthy'
        pred_labels.append(1)

    # st.write(prediction)
    st.write(s)

if(click):
    pred()