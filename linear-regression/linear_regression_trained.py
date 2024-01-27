import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error



# Daten einlesen

# Daten in ein DataFrame laden
df = pd.read_csv("../datasets/possum.csv")

# x-Wert (unabhängige Variable): totlngth
x = df['totlngth'].values.reshape(-1, 1)

# y-Wert (abhängige Variable): hdlngth
y = df['hdlngth'].values

# Aufteilen der Daten in Trainings- und Testsets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Lineare Regression
regression = LinearRegression()
regression.fit(x_train, y_train)

# Vorhersagen für Trainings- und Testdaten
y_train_pred = regression.predict(x_train)
y_test_pred = regression.predict(x_test)

mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

# Visualisierung der Trainingsdaten und der Regression
st.subheader('Trainingsdaten und Lineare Regression')
plt.scatter(x_train, y_train, color='blue', label='Trainingsdaten')
plt.plot(x_train, y_train_pred, color='red', label='Lineare Regression (Trainingsdaten)')
plt.xlabel('totlngth')
plt.ylabel('hdlngth')
plt.legend()
plt.title('Trainingsdaten und Lineare Regression')
st.pyplot(plt)

# Visualisierung der Testdaten und der Regression
st.subheader('Testdaten und Lineare Regression')
plt.scatter(x_test, y_test, color='blue', label='Testdaten')
plt.plot(x_test, y_test_pred, color='red', label='Lineare Regression (Testdaten)')
plt.xlabel('totlngth')
plt.ylabel('hdlngth')
plt.legend()
plt.title('Testdaten und Lineare Regression')
st.pyplot(plt)

# Anzeige der MAE
st.write('**Durchschnittliche Abweichung (MAE) für Trainingsdaten:**', f'{mae_train:.2f}')
st.write('**Durchschnittliche Abweichung (MAE) für Testdaten:**', f'{mae_test:.2f}')

# Anzeigen der Daten in einer Tabelle
st.write('**Datenübersicht:**')
st.write(df)
