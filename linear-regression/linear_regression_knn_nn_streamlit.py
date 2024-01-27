import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

st.title("Visualisierung mit Prediction von unterschiedlichen Modellen")

# Daten einlesen

# Daten in ein DataFrame laden
df = pd.read_csv("../datasets/possum.csv")

method = st.selectbox("Wähle ein Modell:", ["Lineare Regression", "KNN", "NN"])

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
elif method == "KNN":
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

else :
    def build_model(input_size,
                num_layers=2,
                layer_sizes=[8, 10],
                dropout_rate=0.1,
                loss_function='mean_squared_error',
                learning_rate=0.001):
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(input_shape=(input_size,)))

        for i in range(num_layers):
            model.add(keras.layers.Dense(layer_sizes[i], activation='relu'))
            model.add(keras.layers.Dropout(dropout_rate))

        model.add(keras.layers.Dense(1))

        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=loss_function, metrics=['mae'])
        return model
    
    X = df[['totlngth', 'skullw']]
    y = df['hdlngth']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train.shape

    nn = build_model(input_size=X_train.shape[1], learning_rate=0.0001, num_layers=1)
    history = nn.fit(X_train_scaled, y_train, epochs=200, validation_data=(X_test_scaled, y_test), verbose=0)

    def plot_learning_curve(history, model_name):
        plt.plot(history.history['mae'])
        plt.plot(history.history['val_mae'])
        plt.title(model_name + ' - Model Mean Absolute Error')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend(['Train', 'Test'], loc='upper right')
        plt.show()

        st.pyplot(plt)

        st.header("NN Lernkurve")


    def predict_values(user_input, user_input_2):
        scaled_user_input = scaler.transform([[user_input, user_input_2]])
        predicted_value = nn.predict(scaled_user_input)
        st.write("Die Kopflenge für die eingegebenen Daten ist: {}".format(predicted_value))

    plot_learning_curve(history, "default model")

    form = st.form(key="user_form")
    input1 = form.number_input("Gebe einen Wert für totlength ein:", value=5)
    input2 = form.number_input("Gebe einen Wert für skullw ein:", value=5)

    if form .form_submit_button("Start Prediction"):
        predict_values(input1, input2)

    # user_input = st.number_input("Gebe einen Wert für totlength ein:", on_change=None)
    # user_input_2 = st.number_input("Gebe einen Wert für skullw ein:", on_change=None)
    # st.button("Start Prediction", key=None, help=None, on_click=lambda: predict_values(user_input, user_input_2))
    