import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Lade den Datensatz
df = pd.read_csv("../datasets/possum.csv")

# Optionen für das Modell
model_options = ['Lineare Regression', 'K-nearest Neighbors']
selected_model = st.sidebar.selectbox('Wähle ein Modell:', model_options)

# Streamlit App
st.title('Modellvergleich: Lineare Regression vs. K-nearest Neighbors')

if selected_model == 'Lineare Regression':
    # x-Wert (unabhängige Variable): totlngth
    x = df['totlngth'].values.reshape(-1, 1)
    # y-Wert (abhängige Variable): hdlngth
    y = df['hdlngth'].values

    # Lineare Regression
    regression = LinearRegression()
    regression.fit(x, y)

    # Vorhersagen
    y_pred = regression.predict(x)

    # Visualisierung der Daten und der Regression
    st.write('### Visualisierung der Daten und der Regression (Lineare Regression)')
    fig, ax = plt.subplots()
    ax.scatter(x, y, color='blue', label='Daten')
    ax.plot(x, y_pred, color='red', label='Lineare Regression')
    ax.set_xlabel('totlngth')
    ax.set_ylabel('hdlngth')
    ax.legend()
    st.pyplot(fig)

    # Details zur Regression
    st.write('### Details zur Linearen Regression')
    st.write('Steigung (Regression Coefficient):', regression.coef_[0])
    st.write('Achsenabschnitt (Intercept):', regression.intercept_)

elif selected_model == 'K-nearest Neighbors':
    # Wähle die relevanten Features für KNN
    features = ['totlngth', 'hdlngth']
    X = df[features]

    # Initialisiere den KNN-Algorithmus
    knn = NearestNeighbors(n_neighbors=5, algorithm='auto')

    # Passe das Modell an
    knn.fit(X)

    # Visualisierung der K-nearest Neighbors
    st.write('### Visualisierung der K-nearest Neighbors')
    totlngth_input = st.sidebar.slider('totlngth', min_value=float(df['totlngth'].min()), max_value=float(df['totlngth'].max()), value=float(df['totlngth'].min()))
    hdlngth_input = st.sidebar.slider('hdlngth', min_value=float(df['hdlngth'].min()), max_value=float(df['hdlngth'].max()), value=float(df['hdlngth'].min()))

    # Finde die K-nearest Neighbors für die ausgewählte Eingabe
    input_data = np.array([totlngth_input, hdlngth_input]).reshape(1, -1)
    distances, indices = knn.kneighbors(input_data)

    # Zeige die K-nearest Neighbors
    st.write('### K-nearest Neighbors:')
    st.write(df.iloc[indices[0]])

    # Visualisierung der ausgewählten Features
    fig, ax = plt.subplots()
    # Scatter-Plot für K-nearest Neighbors
    ax.scatter(df.iloc[indices[0]]['totlngth'], df.iloc[indices[0]]['hdlngth'], color='blue', label='K-nearest Neighbors')
    # Eingabe als roter Punkt
    ax.scatter(totlngth_input, hdlngth_input, color='red', label='Eingabe')
    ax.set_xlabel('totlngth')
    ax.set_ylabel('hdlngth')
    ax.legend()
    st.pyplot(fig)
