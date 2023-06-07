import streamlit as st
import numpy as np
from pandas import DataFrame,read_csv
import pickle
from tensorflow.keras.models import load_model
from datetime import datetime
from math import exp

X_train = read_csv('Regression_Models/Used_Car_Price_Prediction/datasets/X_train.csv')
X_train = X_train.drop('carID',axis=1)

brand_values = X_train.brand.unique()
transmission_values = tuple(X_train.transmission.unique())
fuelType_values = tuple(X_train.fuelType.unique())
model_values = tuple(pickle.load(open('most_freq_models', 'rb')))
preprocessing_data = pickle.load(open('preprocessing_data', 'rb'))
model = load_model('used_car_price_model.h5') 

def prep_data(x):
    
    x = DataFrame([x],columns=X_train.columns)
    x.insert(0,'years',year_now()-x['year'])
    x = x.drop('year',axis=1)
    x = preprocessing_data[0].transform(x)
    x = preprocessing_data[1].transform(x)
    x = x[:,:10]
    x = x.reshape(1,10)
    
    return x
    

def compute_price(x):
    
    y = model.predict(x)
    y = float(y)
    y = exp(y)
    y = round(y,2)
    
    return y

def year_now():
    
    now = datetime.now()
    
    return now.year

def main():
    
    x = []
    
    st.title('Used Car Price Calculator')

    x.append(st.selectbox('Brand',brand_values))
    x.append(st.selectbox('Model',model_values))
    x.append(st.slider('Year',1950,year_now(),step=1))
    x.append(st.selectbox('Transmission',transmission_values))
    x.append(st.slider('Mileage',0,100000,step=1))
    x.append(st.selectbox('FuelType',fuelType_values))
    x.append(st.slider('Tax',0,1000,step=1))
    x.append(st.slider('MPG',0.0,1000.0,step=0.1))
    x.append(st.slider('EngineSize',0.0,10.0,step=0.1))

    compute_button = st.button('Compute Price')

    if compute_button:

        x = prep_data(x)
        y = compute_price(x)

        st.write('The price car is : ${:,.2f}'.format(y))


if __name__ == '__main__':
    
    main()
    



