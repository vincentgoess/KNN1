from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Daten einlesen
file_path = "/Users/student-sbs/PycharmProjects/KNN1/winequality-red.csv"  # Datei muss im aktuellen Verzeichnis sein
df = pd.read_csv(file_path, delimiter=';')

# 2. Erste und letzte Zeilen des Dataframe
print(f"Erste Zeilen: \n{df.head()} \nLetzte Zeilen: \n {df.tail()}")

# 3. Zusammenfassung (Datentypen, Anzahl der Nicht-Null-Einträge,...)
print(df.info())

# 4. alcohol und pH erste 10 Zeilen
print(df['alcohol'].head(10), df['pH'].head(10))

# 5. Daten mit quality genau 8
print(df[df['quality'] == 8])

# 6. Daten mit mehreren Bedingungen anzeigen
print(df[(df['alcohol'] > 12.5) & (df['quality'] >= 7)])

# 7. Neue Spalte aus Verrechnung zweier Spalten hinzufügen (Dichte/Alkohol - Rate)
df['density_alcohol_ratio'] = df['density'] / df['alcohol']
print(df['density_alcohol_ratio'].head())

# 8. Werte innerhalb einer Spalte ändern

def quality_label(q):
    if q == 3:
        return "sehr schlecht"
    elif q == 4:
        return "schlecht"
    elif q == 5:
        return "okay"
    elif q == 6:
        return "gut"
    else:
        return "sehr gut"

df['quality_label'] = df['quality'].apply(quality_label)
print("Klassifizierte quality-Werte:\n",df[['quality','quality_label']]) # Klassifizierung der Werte

# 9. pH-Werte unter 3.0 entfernen
print(df[df['pH'] >= 3.0])

# 10. Spaltenüberschriften
print(df.columns)

# 11. Spaltentitel übersetzen
deutsch_columns = {
    'fixed acidity': 'fester Säuregehalt',
    'volatile acidity': 'flüchtiger Säuregehalt',
    'citric acid': 'Zitronensäure',
    'residual sugar': 'Restzucker',
    'chlorides': 'Chloride',
    'free sulfur dioxide': 'freies Schwefeldioxid',
    'total sulfur dioxide': 'Gesamtschwefeldioxid',
    'density': 'Dichte',
    'pH': 'pH-Wert',
    'sulphates': 'Sulfate',
    'alcohol': 'Alkohol',
    'quality': 'Qualität'
}
df.rename(columns=deutsch_columns, inplace=True)
print("\nErste Zeilen nach Umbenennung der Spalten:")
print(df.head())

# 12. Visualisierung (Scatterplot von Alkohol vs. Qualität)
print("\nErstelle Scatterplot für Alkohol vs. Qualität...")
sns.scatterplot(x=df['Alkohol'], y=df['Qualität'])
plt.xlabel("Alkoholgehalt")
plt.ylabel("Qualität")
plt.title("Alkohol vs. Qualität")
plt.show()

# 13. KMeans-Clustering
# Entfernen der Spalten 'quality' und 'quality_label', da diese nicht für das Clustering verwendet werden
df_clustering = df.drop(columns=['Qualität', 'quality_label'])

# KMeans mit 3 Clustern anwenden
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(df_clustering)

# Visualisierung der Cluster
print("\nCluster Visualisierung: Alkohol vs. Dichte (mit Cluster-Zugehörigkeit)")
sns.scatterplot(x=df['Alkohol'], y=df['Dichte'], hue=df['cluster'], palette='viridis')
plt.xlabel("Alkoholgehalt")
plt.ylabel("Dichte")
plt.title("KMeans Clustering: Alkohol vs. Dichte")
plt.show()

# Cluster-Zugehörigkeit anzeigen
print("\nCluster-Zugehörigkeit für die ersten 10 Zeilen:")
print(df[['Alkohol', 'Dichte', 'cluster']].head(10))

'''	•	Der Scatterplot zeigt die Verteilung der Weindaten in Bezug auf Alkoholgehalt und Dichte.
	•	Das KMeans-Clustering unterteilt diese Weine basierend auf diesen Merkmalen in 3 Cluster, die 
	durch Farben im Scatterplot hervorgehoben werden. Dadurch wird ersichtlich, wie sich die Weine in 
	diese Cluster gruppieren, z. B. in Cluster mit hohem oder niedrigem Alkoholgehalt und Dichte.'''
