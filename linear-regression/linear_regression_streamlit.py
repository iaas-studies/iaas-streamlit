import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

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

# Dropdown-Menü für y-Achse
y_column = st.selectbox("Wähle eine Spalte für die y-Achse:", numeric_columns)

# Lineare Regression durchführen
X = df[x_column].values.reshape(-1, 1)
y = df[y_column].values.reshape(-1, 1)

model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)

# Ergebnisse anzeigen
plt.scatter(X, y, color='blue', label='Originaldaten')
plt.plot(X, predictions, color='red', linewidth=2, label='Lineare Regression')
plt.xlabel(x_column)
plt.ylabel(y_column)
plt.legend()

st.pyplot(plt)

# Vorhersage (Prediction) für benutzerdefinierten x-Wert
st.header("Vorhersage (Prediction)")
user_input = st.number_input("Gebe einen Wert für {} ein:".format(x_column))
predicted_value = model.predict([[user_input]])
st.write("Die vorhergesagte {} für den eingegebenen {}-Wert ist: {}".format(y_column, x_column, predicted_value[0][0]))
