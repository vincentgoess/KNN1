import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Excel-Datei einlesen (benötigt: openpyxl!)
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
    df["Kategorie"].value_counts().plot(kind="barh", title="Häufigkeit nach Kategorie") #kind="bar" für Säulendiagramm
    plt.tight_layout()
    plt.show()

# 9. Entscheidungsbaum: Vorhersage einer Zielspalte "Ziel"
if "Ziel" in df.columns:
    X = df.drop(columns=["Ziel"])
    y = df["Ziel"]

    # Nur numerische Spalten verwenden
    X = X.select_dtypes(include=["int64", "float64"])

    # Aufteilen in Trainings- und Testdaten
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\nEntscheidungsbaum-Vorhersage – Genauigkeit:")
    print(accuracy_score(y_test, y_pred))

# 10. Vorhersage für nicht-numerische Zielspalte mit Label-Encoding
if "Ziel" in df.columns:
    # Falls Zielspalte Text enthält → in Zahlen umwandeln
    if df["Ziel"].dtype == "object":
        df["Ziel"] = df["Ziel"].astype("category").cat.codes

    X = df.drop(columns=["Ziel"])
    y = df["Ziel"]

    # Nur numerische Eingaben verwenden
    X = X.select_dtypes(include=["int64", "float64"])

    # Trainings-/Testdaten
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\n/10 Vorhersage für nicht-numerische Zielspalte (encodiert):")
    print("Genauigkeit:", accuracy_score(y_test, y_pred))






# Daten löschen
    # Zeilen mit bestimmtem Wert entfernen (z. B. "?")
    df = df[df["Spalte1"] != "?"]  # behält nur Zeilen, wo Spalte1 nicht "?" ist

    # Zeilen mit fehlenden Werten löschen (NaN)
    df = df.dropna()  # alle Zeilen mit NaN in beliebiger Spalte
    # Nur wenn NaN in bestimmter Spalte:
    df = df.dropna(subset=["Spalte1"])

    # Ganze Spalte löschen
    df = df.drop(columns=["Spalte1"])

    # Einzelne Zeile per Index löschen
    df = df.drop(index=3)  # löscht Zeile mit Index 3
