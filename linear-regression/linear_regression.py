import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Daten einlesen

# Daten in ein DataFrame laden
df = pd.read_csv("../datasets/possum.csv")

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
plt.show()