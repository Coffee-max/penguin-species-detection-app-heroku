import streamlit as st
import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import *
from PIL import Image

st.set_page_config(layout="wide")
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: black;'>Penguin Prediction App</h1>", unsafe_allow_html=True)
col1 = st.sidebar
col2, col3 = st.columns((2,1))

image = Image.open('pg-logo.jpg')

col3.image(image, width = 400)


#-------------------#
# About
expander_bar = col2.expander("About")
expander_bar.markdown("""
* **Python libraries:** pandas, streamlit, numpy, seaborn, pickle, matplotlib, sklearn.
* This app predicts the **Palmer Penguin** species!
**Note**:Data obtained from the [palmerpenguins library](https://github.com/allisonhorst/palmerpenguins) in R by Allison Horst.
""")


col1.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

# Collects user input features into dataframe
uploaded_file = col1.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island = col1.selectbox('Island',('Biscoe','Dream','Torgersen'))
        sex = col1.selectbox('Sex',('male','female'))
        bill_length_mm = col1.slider('Bill length (mm)', 32.1,59.6,43.9)
        bill_depth_mm = col1.slider('Bill depth (mm)', 13.1,21.5,17.2)
        flipper_length_mm = col1.slider('Flipper length (mm)', 172.0,231.0,201.0)
        body_mass_g = col1.slider('Body mass (g)', 2700.0,6300.0,4207.0)
        data = {'island': island,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

url = 'https://raw.githubusercontent.com/dataprofessor/streamlit_freecodecamp/main/app_8_classification_penguins/penguins_cleaned.csv'

penguins_raw = pd.read_csv(url)
penguins = penguins_raw.drop(columns=['species'],axis=1)
df = pd.concat([input_df,penguins],axis=0)
# Encoding of ordinal features

encode = ['sex','island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)
col2.subheader('User Input features')

if uploaded_file is not None:
    col2.write(df)
else:
    col2.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    col2.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('C:/Users/samar/Desktop/Python/Streamlit DS Projects/Penguin EDA/pg_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)



col2.button('Prediction')
penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
col2.dataframe(penguins_species[prediction], width=200)

col2.button('Prediction Probability')
col2.dataframe(prediction_proba)

if col2.button('Scatterplot Matrix'):
    col2.header('Scatterplot Matrix - Spread and Correlation')
    df = pd.read_csv(url)
    sns.set_theme(style="ticks")
    sns.pairplot(df, hue="species")
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(7, 5))
        ax = sns.pairplot(df, hue="species")
    col2.pyplot()