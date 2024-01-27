import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

# Funktion zum Überprüfen des Datentyps der Spalten
def is_numeric_dtype(column):
    return pd.api.types.is_numeric_dtype(column)

# Streamlit App
st.title("Lineare Regression Visualisierung mit Prediction")


# Daten aus der CSV-Datei lesen
df = pd.read_csv("../datasets/possum.csv")
numeric_columns = [col for col in df.columns if is_numeric_dtype(df[col])]


# Dropdown-Menü für x-Achse
x_column = st.selectbox("Wähle eine Spalte für die x-Achse:", numeric_columns)

# Dropdown-Menü für x-Achse
x_column_2 = st.selectbox("Wähle eine Spalte für die 2. x-Achse:", numeric_columns)

# Dropdown-Menü für y-Achse
y_column = st.selectbox("Wähle eine Spalte für die y-Achse:", numeric_columns)

# Dropdown-Menü für Methode
method = st.selectbox("Wähle eine Methode:", ["Lineare Regression", "k-Nearest Neighbors"])

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



