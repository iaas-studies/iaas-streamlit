import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import streamlit as st

st.title("Lineare Regression Visualisierung mit Prediction")

# Daten einlesen

# Daten in ein DataFrame laden
df = pd.read_csv("../datasets/possum.csv")

method = st.selectbox("Wähle eine Spalte für die y-Achse:", ["Lineare Regression", "KNN"])

if method == "Lineare Regression":
    # x-Wert (unabhängige Variable): site
    x = df['totlngth'].values.reshape(-1, 1)

    # y-Wert (abhängige Variable): skullw
    y = df['hdlngth'].values

    # Lineare Regression
    regression = LinearRegression()
    regression.fit(x, y)

    # Vorhersagen
    y_pred = regression.predict(x)

    # Visualisierung der Daten und der Regression
    plt.scatter(x, y, color='blue', label='Daten')
    plt.plot(x, y_pred, color='red', label='Lineare Regression')
    plt.xlabel('totlngth')
    plt.ylabel('hdlngth')
    plt.legend()

    st.pyplot(plt)

    st.header("Vorhersage (Prediction)")
    user_input = st.number_input("Gebe einen Wert für totlength ein:")
    predicted_value = regression.predict([[user_input]])
    st.write("Die vorhergesagte hdlngth für den eingegebenen totlngth-Wert ist: {}".format(predicted_value[0]))
else :
    # x-Wert (unabhängige Variable): site
    x = df[['totlngth', 'skullw']]

    # y-Wert (abhängige Variable): skullw
    y = df['hdlngth']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    def predict(k, x_train, p_oi):
        distance = []

        for i in range(x_train.shape[0]):
            point_in_train = x_train.iloc[i]
            dist_to_point_i = np.sqrt((p_oi[0] - point_in_train[0])**2+(p_oi[1] - point_in_train[1]**2))
            distance.append((dist_to_point_i, i))

        knn = sorted(distance, key=lambda x: x[0])
        idxs = [entry[1] for entry in knn][:k]
        xs = [pd.DataFrame(x_train['totlngth']).iloc[idx] for idx in idxs]
        xs_2 = [pd.DataFrame(x_train['skullw']).iloc[idx] for idx in idxs]
        ys = [pd.DataFrame(y_train).iloc[idx] for idx in idxs]

        # # Visualisierung der Daten und der Regression
        plt.scatter(xs, xs_2, color='blue', label='Näheste Nachbarn')
        plt.scatter(p_oi[0], p_oi[1], color='red', label='Eingabe')
        plt.xlabel('totlngth')
        plt.ylabel('skullw')
        plt.legend()

        st.pyplot(plt)

        st.header("Vorhersage (Prediction)")

        return np.mean(ys)

    user_input = st.number_input("Gebe einen Wert für totlength ein:")
    user_input_2 = st.number_input("Gebe einen Wert für skullw ein:")
    predicted_value = predict(5, pd.DataFrame(x_train), [[user_input], [user_input_2]])
    st.write("Die vorhergesagte hdlngth für den eingegebenen totlngth-Wert ist: {}".format(predicted_value))
