import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. CSV-Datei einlesen
df = pd.read_csv("beispiel.csv", delimiter=",")

# 2. Erste und letzte Spalten anzeigen
print("Erste Spalte:")
print(df.iloc[:, 0].head())
print("\nLetzte Spalte:")
print(df.iloc[:, -1].head())

# 3. Zusammenfassung der Daten
print("\nZusammenfassung:")
print(df.info())

# 4. Spaltennamen ausgeben
print("\nSpaltennamen:")
print(df.columns)

# 5. Daten gruppieren (z. B. nach Kategorie und Mittelwert)
if "Kategorie" in df.columns:
    print("\nGruppiert nach Kategorie (Mittelwert):")
    print(df.groupby("Kategorie").mean(numeric_only=True))

# 6. Daten filtern, z. B. nur Zeilen mit Wert > 100 in Spalte "Wert"
if "Wert" in df.columns:
    print("\nGefilterte Zeilen (Wert > 100):")
    print(df[df["Wert"] > 100])

# 7. Neue Spalte erstellen: z. B. Summe aus zwei Spalten
if "A" in df.columns and "B" in df.columns:
    df["Summe"] = df["A"] + df["B"]
    print("\nNeue Spalte 'Summe':")
    print(df["Summe"].head())

# 8. Balkendiagramm: z. B. Häufigkeit von Kategorien
if "Kategorie" in df.columns:
    df["Kategorie"].value_counts().plot(kind="barh", title="Häufigkeit nach Kategorie")  # kind="bar" für Säulen
    plt.tight_layout()
    plt.show()

# 9. Vorhersage einer Zielspalte (wenn Ziel numerisch)
if "Ziel" in df.columns and df["Ziel"].dtype != "object":
    X = df.drop(columns=["Ziel"])
    y = df["Ziel"]
    X = X.select_dtypes(include=["int64", "float64"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\n/9 Entscheidungsbaum-Vorhersage – Genauigkeit:")
    print(accuracy_score(y_test, y_pred))

# 10. Vorhersage für nicht-numerische Zielspalte (Label-Encoding)
if "Ziel" in df.columns and df["Ziel"].dtype == "object":
    df["Ziel"] = df["Ziel"].astype("category").cat.codes

    X = df.drop(columns=["Ziel"])
    y = df["Ziel"]
    X = X.select_dtypes(include=["int64", "float64"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\n/10 Vorhersage für nicht-numerische Zielspalte (encodiert):")
    print("Genauigkeit:", accuracy_score(y_test, y_pred))

# 🔻 Daten löschen (optional):
# Beispielhafte Spalte 'Spalte1' – muss im CSV vorhanden sein
if "Spalte1" in df.columns:
    # Zeilen mit bestimmtem Wert entfernen (z. B. "?")
    df = df[df["Spalte1"] != "?"]

    # Zeilen mit NaN (Not a Number) löschen
    df = df.dropna()  # oder gezielt mit: df.dropna(subset=["Spalte1"])

    # Ganze Spalte löschen
    df = df.drop(columns=["Spalte1"])

    # Zeile nach Index löschen (z. B. 3)
    if 3 in df.index:
        df = df.drop(index=3)
