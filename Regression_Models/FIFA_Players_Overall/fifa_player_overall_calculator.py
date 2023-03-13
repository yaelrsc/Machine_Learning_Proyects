import streamlit as st
import numpy as np
import pickle

from tensorflow.keras.models import load_model

col_abil_names = ['Player information','ATTACKING','SKILL','MOVEMENT',
                  'POWER','MENTALITY','DEFENDING','GOALKEEPING']

model_gk = load_model('Regression_Models/FIFA_Players_Overall/Models/fifa_gk_model.h5')
model_def = load_model('Regression_Models/FIFA_Players_Overall/Models/fifa_def_model.h5')
model_med = load_model('Regression_Models/FIFA_Players_Overall/Models/fifa_med_model.h5')
model_del = load_model('Regression_Models/FIFA_Players_Overall/Models/fifa_del_model.h5')
scalers = pickle.load(open('Models/FIFA/scalers', 'rb'))

def predict(x,pos):
    
    if pos == 'GK':
        
        x = scalers[0][0].transform(x)
        y = model_gk.predict(x)
        y = scalers[0][1].inverse_transform(y)
        
    if pos == 'DEF':
        
        x = scalers[1][0].transform(x)
        y = model_def.predict(x)
        y = scalers[1][1].inverse_transform(y)
        
    if pos == 'MED':
        
        x = scalers[2][0].transform(x)
        y = model_med.predict(x)
        y = scalers[2][1].inverse_transform(y)
        
    if pos == 'ST':
        
        x = scalers[3][0].transform(x)
        y = model_del.predict(x)
        y = scalers[3][1].inverse_transform(y)
    
    
    y = int(y)
    
    return y

def main():
    
    x = np.zeros(34)

    st.title('FIFA Player Overall Calculator')


    st.header(col_abil_names[0])

    st.text_input('Name')

    st.text_input('Nationality')

    st.text_input('Age')

    pos = st.selectbox('Position:', ['GK','DEF','MED','ST'])


    with st.container():

        col_abilities = st.columns(3,gap='medium')

        with col_abilities[0]:

            st.header(col_abil_names[1])
            x[0] = st.slider('Crossing',0,99)
            x[1] = st.slider('Finishing',0,99)
            x[2] = st.slider('Heading Accuracy',0,99)
            x[3] = st.slider('Short Passing',0,99)
            x[4] = st.slider('Volleys',0,99)

        with col_abilities[1]:

            st.header(col_abil_names[2])
            x[5] = st.slider('Dribbling',0,99)
            x[6] = st.slider('Curve',0,99)
            x[7] = st.slider('FK Accuracy',0,99)
            x[8] = st.slider('Long Passing',0,99)
            x[9] = st.slider('Ball Control',0,99)

        with col_abilities[2]:

            st.header(col_abil_names[3])
            x[10] = st.slider('Acceleration',0,99)
            x[11] = st.slider('Sprint Speed',0,99)
            x[12] = st.slider('Agility',0,99)
            x[13] = st.slider('Reactions',0,99)
            x[14] = st.slider('Balance',0,99)





    with st.container():

        col_abilities = st.columns(3,gap='medium')

        with col_abilities[0]:

            st.header(col_abil_names[4])
            x[15] = st.slider('Shot Power',0,99)
            x[16] = st.slider('Jumping',0,99)
            x[17] = st.slider('Stamina',0,99)
            x[18] = st.slider('Strength',0,99)
            x[19] = st.slider('Long Shots',0,99)

        with col_abilities[1]:

            st.header(col_abil_names[5])
            x[20] = st.slider('Aggression',0,99)
            x[21] = st.slider('Interceptions',0,99)
            x[22] = st.slider('Positioning',0,99)
            x[23] = st.slider('Vision',0,99)
            x[24] = st.slider('Penalties',0,99)
            x[25] = st.slider('Composure',0,99)

        with col_abilities[2]:

            st.header(col_abil_names[6])
            x[26] = st.slider('Defensive Awareness',0,99)
            x[27] = st.slider('Standing Tackle',0,99)
            x[28] = st.slider('Sliding Tackle',0,99)


    with st.container():

        col_abilities = st.columns(2,gap='medium')

        with col_abilities[0]:

            st.header(col_abil_names[7])
            x[29] = st.slider('GK Diving',0,99)
            x[30] = st.slider('GK Handling',0,99)
            x[31] = st.slider('GK Kicking',0,99)
            x[32] = st.slider('GK Positioning',0,99)
            x[33] = st.slider('GK Reflexes',0,99)

        with col_abilities[1]:

            st.header('Overall Rating')
            calculate = st.button('Calculate')

            if calculate:

                x = x.reshape(1,34)

                y = predict(x, pos=pos)

                st.write('Player Overall: {}'.format(y))
                

if __name__ == '__main__':
    
    main()


