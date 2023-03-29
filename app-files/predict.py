import streamlit as st
import numpy as np
import pandas as pd
import pickle

with open('features.pkl', 'rb') as fg:
    feat = pickle.load(fg)
    X = pd.DataFrame(feat)
with open('Delhi_HousePrice_Prediction.pkl', 'rb') as f:
    lrm = pickle.load(f)
with open('addresses.pkl', 'rb') as fb:
    ad = pickle.load(fb)

def predict_price(area, bed, bath, balc, loc, typb):
    loc_ind = np.where(X.columns == loc)[0]
    typ_ind = np.where(X.columns == typb)[0]
    h = np.zeros(len(X.columns))
    h[0] = area
    h[1] = bed
    h[2] = bath
    h[3] = balc
    if loc_ind >= 0:
        if typ_ind >= 0:
            h[typ_ind] = 1
            h[loc_ind] = 1
        h[loc_ind] = 1
    if typ_ind >= 0:
        h[typ_ind] = 1

    return lrm.predict([h])

def show_predictpage():
    st.title('Welcome to House Price Predictor')
    st.caption('**:blue[Predict Housing Prices in NCR]**')

    loc = st.selectbox(
        '**Enter a location in NCR**',
        ad)
    area = st.text_input('**Area in sq-ft**', 'Enter a value')
    bed = st.number_input('**Bedrooms**', min_value=0, max_value=20, step=1)
    bath = st.number_input('**Bathrooms**', min_value=0, max_value=20, step=1)
    balc = st.number_input('**Balcony**', min_value=0, max_value=20, step=1)
    typb = st.radio(
        "**Type of Building**",
        ('Flat', 'Individual House'))

    if st.button('Predict Price'):
        p = predict_price(area, bed, bath, balc, loc, typb)
        st.success(f'Price of the property is INR {p[0]:.0f}')
