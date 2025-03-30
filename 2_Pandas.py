import pandas as pd


def mylog(msg):
    print("*" * 50)
    print(msg)
    print("*" * 50)


mylog("Datei einlesen")
df = pd.read_csv("/Users/student-sbs/PycharmProjects/KNN1/iris.csv")

mylog("Erste und letzten 5 Zeilen ausgeben")
print(df.head(5), "\n" + "-" * 50 + "\n", df.tail(5))
mylog("CSV Zusammenfassung")
print(df.info())

# FILTER
mylog("ERSTE 10 ZEILEN MIT SPALTENFILTER")
filter = df[['sepal.length', 'sepal.width']] # Spaltenfilter
print(filter.head(10)) # Erste 10 Zeilen davon

mylog("REIHENFILTER")
rowfilter = df[df['species'] == 'Setosa'] # Reihenfilter
print(rowfilter)
mylog("rowfilterung mit nummern")
rowfilter_number = df[df["petal.length"] > 1.5] # Reihen rausfiltern mit Wert größer 1.5
print(rowfilter_number)

# ADD COLUMN, REPLACE VALUES, DELETE ROWS
mylog("ADD COLUMN")
df["sepal.area"] = df["sepal.length"] * df["sepal.width"] # Spalte hinzufügen
print(df.head(10))

mylog("WERTE ERSETZEN")
# Anzeige ändern
print(df['species'].map({'Setosa': 'S', 'Versicolor': 'Ve', 'Virginica': 'Vi'})) # Werte innerhalb 'species' ändern
# Werte komplett ersetzen
df["species"] = df["species"].replace(regex='Setosa', value='S') # regex='Setosa' soll ersetzt werden mit value='S'
df["species"] = df["species"].replace(regex='Versicolor', value='Ve') # regex_'Versicolor' soll ersetzt werden mit value='Ve'
df["species"] = df["species"].replace(regex='Virginica', value='Vi') # regex='Virginica' soll ersetzt werden mit value='Vi'
print(df)

mylog("DELETE ROWS")
rowfilter_for_delete = df[df["sepal.length"] < 4.5] # Reihen löschen
print(rowfilter_for_delete)

