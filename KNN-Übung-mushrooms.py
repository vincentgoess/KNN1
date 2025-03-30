import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Geben Sie die ersten Zeilen aus. Achten Sie auf das korrekte Trennzeichen.
path = "/Users/student-sbs/PycharmProjects/KNN1/mushrooms.csv"
data = pd.read_csv(path, delimiter=',')
print(data.head())  # Ausgabe der ersten 5 Zeilen

# 2. Ermitteln Sie die Anzahl der leeren Zellen.
print("Anzahl der leeren Zellen:", data.isnull().sum())

# 3. Teilen Sie die Daten in zwei Tabellen auf. Zielspalte OHE-kodiert.
col_name = 'class'  # Zielspalte (Pilzklassifikation)
col = pd.get_dummies(data[col_name], dtype=float)  # OHE der Zielspalte
data = data.drop([col_name], axis=1)  # Entfernen der Zielspalte aus den Eingabedaten

# 4. Transformieren Sie alle restlichen Spalten mit dem LabelEncoder.
le = LabelEncoder()
data = data.apply(le.fit_transform)  # Alle Spalten mit LabelEncoder transformieren

# 5. Ermitteln Sie nun die Korrelationen.
correlations = data.corr()  # Berechnung der Korrelationen
print(correlations)  # Ausgabe der Korrelationen

# 6. Skalieren der Daten
s_scaler = StandardScaler()
data = s_scaler.fit_transform(data)  # Skalierung der Daten

# 7. KNN aufbauen, trainieren und testen
train_data, test_data, train_col, test_col = train_test_split(data, col, test_size=0.2, random_state=42)

# Aufbau des KNN-Modells
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(data.shape[1],)))  # Eingabeschicht angepasst an die Anzahl der Merkmale
model.add(tf.keras.layers.Dense(32, activation=tf.nn.sigmoid))  # Erste Hidden-Layer mit 32 Neuronen
model.add(tf.keras.layers.Dense(64, activation=tf.nn.sigmoid))  # Zweite Hidden-Layer mit 64 Neuronen
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))  # Ausgabeschicht mit 2 Neuronen (essbar oder giftig)

# Konfiguration des Lernprozesses
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modell trainieren (30 Epochen)
model.fit(train_data, train_col, epochs=30)

# 8. Testen des Modells
test_loss, test_acc = model.evaluate(test_data, test_col)
print('Test accuracy:', test_acc)  # Ausgabe der Testgenauigkeit

# 9. Hyperparameter anpassen und testen
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(data.shape[1],)))  # Eingabeschicht angepasst
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))  # ReLU Aktivierung und mehr Neuronen
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # Weitere Neuronen
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))  # Zwei Klassen (essbar, giftig)

# Modell kompilieren und trainieren
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, train_col, epochs=50)  # Mehr Epochen

# Evaluierung der Genauigkeit nach Anpassung
test_loss, test_acc = model.evaluate(test_data, test_col)
print('Test accuracy nach Hyperparameter-Anpassung:', test_acc)
