import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
import streamlit as st
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras


# Funktion zum Überprüfen des Datentyps der Spalten
def is_numeric_dtype(column):
    return pd.api.types.is_numeric_dtype(column)

# Streamlit App
st.title("Lineare Regression Visualisierung mit Prediction")


# Daten aus der CSV-Datei lesen
csv_path = Path(__file__).parents[1] / 'datasets/possum.csv'
df = pd.read_csv(csv_path)

numeric_columns = [col for col in df.columns if is_numeric_dtype(df[col])]

# Dropdown-Menü für Methode
method = st.selectbox("Wähle eine Methode:", ["Lineare Regression", "k-Nearest Neighbors", "Nueronal Network"])

# Dropdown-Menü für x-Achse
x_column = st.selectbox("Wähle eine Spalte für die x-Achse:", numeric_columns)

if method != "Lineare Regression":  
    # Dropdown-Menü für x-Achse
    x_column_2 = st.selectbox("Wähle eine Spalte für die 2. x-Achse:", numeric_columns)


# Dropdown-Menü für y-Achse
y_column = st.selectbox("Wähle eine Spalte für die y-Achse:", numeric_columns)

if method == "Lineare Regression":        
    # Lineare Regression durchführen
    X = df[x_column].values.reshape(-1, 1)
    y = df[y_column].values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)

    # Ergebnisse anzeigen
    plt.scatter(X, y, color='blue', label='Originaldaten')
    plt.plot(X, predictions, color='red', linewidth=2, label='Erwartete Daten')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.legend()

    st.pyplot(plt)

    # Vorhersage (Prediction) für benutzerdefinierten x-Wert
    st.header("Vorhersage (Prediction)")
    user_input = st.number_input("Gebe einen Wert für {} ein:".format(x_column))
    predicted_value = model.predict([[user_input]])
    st.write("Die vorhergesagte {} für den eingegebenen {}-Wert ist: {}".format(y_column, x_column, predicted_value[0][0]))

elif method == "k-Nearest Neighbors":
    # k-Nearest Neighbors durchführen
    X = df[[x_column, x_column_2]].values
    y = df[y_column].values.reshape(-1, 1)

    # Anzahl der nächsten Nachbarn festlegen (z.B., k=4)
    k = 4
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X, y)

    # Ergebnisse anzeigen
    plt.scatter(X[:, 0], y, color='blue', label='Originaldaten')
    plt.scatter(X[:, 1], y, color='black', label='Originaldaten 2')
    plt.xlabel(x_column + " " + x_column_2)
    plt.ylabel(y_column)

    # Vorhersage (Prediction) für benutzerdefinierte x-Werte
    st.header("Vorhersage (Prediction)")
    user_input_x1 = st.number_input("Gebe einen Wert für {} ein:".format(x_column))
    user_input_x2 = st.number_input("Gebe einen Wert für {} ein:".format(x_column_2))

    # Die nächsten k Nachbarn für die benutzerdefinierten Werte finden
    distances, indices = model.kneighbors([[user_input_x1, user_input_x2]])

    # Nur die nächsten k Nachbarn hervorheben
    nearest_neighbors_x1 = X[indices[0], 0]
    nearest_neighbors_x2 = X[indices[0], 1]

    plt.scatter(nearest_neighbors_x1, y[indices[0]], color='red', marker='x', label='Nächste Nachbarn', s=100)
    plt.scatter(nearest_neighbors_x2, y[indices[0]], color='red', marker='x', s=100)
    plt.legend()

    # Vorhersage (Prediction) für die benutzerdefinierten x-Werte
    predicted_value = model.predict([[user_input_x1, user_input_x2]])
    plt.scatter(user_input_x1, predicted_value, color='green', marker='o', label='Erwarteter Wert', s=100)

    st.write("Die vorhergesagte {} für die eingegebenen {}-Werte ist: {}".format(y_column, [x_column, x_column_2], predicted_value[0]))

    # Das Diagramm anzeigen
    st.pyplot(plt)

elif method=="Nueronal Network":
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
    
    X = df[[x_column, x_column_2]]
    y = df[y_column]
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
        st.write("Die {} für die eingegebenen Daten ist: {}".format(y_column, predicted_value))

    plot_learning_curve(history, "default model")

    form = st.form(key="user_form")
    input1 = form.number_input("Gebe einen Wert für {} ein:".format(x_column), value=5)
    input2 = form.number_input("Gebe einen Wert für {} ein:".format(x_column_2), value=5)

    if form .form_submit_button("Start Prediction"):
        predict_values(input1, input2)


