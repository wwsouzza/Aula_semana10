import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
transacoes = pd.read_csv( "transacoes.csv" )
X = transacoes.drop(columns=[ "id_transacao" , "fraude"])
y = transacoes[ "fraude"]
X_treino, X_teste, y_treino, y_teste = train_test_split(
 X, y, test_size= 0.25, stratify=y, random_state= 42)
print(y_treino.mean(). round(4), y_teste.mean(). round(4))

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
numericas = [ "valor", "hora", "distancia_residencia_km" , "tentativas_24h" ,
"idade_conta_dias" ]
prep = ColumnTransformer([
 ("num", StandardScaler(), numericas),
 ("cat", OneHotEncoder(), [ "canal"]),
], remainder= "passthrough" )
print(prep.fit_transform(X_treino).shape)

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
modelo = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=1000))])
modelo.fit(X_treino, y_treino)
print("acurácia no teste:", round(modelo.score(X_teste, y_teste), 3))

from sklearn.metrics import classification_report
y_prev = modelo.predict(X_teste)
print(classification_report(y_teste, y_prev, target_names=["Legítima", "Fraude"]))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_teste, y_prev))

#Porque o quantidade de fraudes versus legitima é muito pequena e apresenta um desbalanceamento excessivo


proba = modelo.predict_proba(X_teste)[:, 1]
print(proba[:5].round(3))
print("maior probabilidade prevista:" , proba.max().round(3))
print("transações acima de 0,5:" , (proba > 0.5).sum())

sns.histplot(x=proba, hue=y_teste.map({0: "Legítima", 1:
"Fraude"}), bins=30, stat="count", common_norm=False)
plt.axvline(0.5, color="black", linestyle="--")
plt.yscale("log")
plt.xlabel("probabilidade prevista de fraude (eixo y em escala log)")
plt.savefig("histograma_probabilidades.png")
plt.close()

from sklearn.metrics import precision_score, recall_score
print("limiar precision recall alertas" )
for corte in [0.5, 0.3, 0.2, 0.1, 0.05]:
 prev = (proba >= corte).astype( int)
 p, r = precision_score(y_teste, prev), recall_score(y_teste, prev)
 print(f"{corte:6.2f} {p:9.2f} {r:6.2f} {prev.sum():7d}" )

from sklearn.tree import DecisionTreeClassifier
arvore = Pipeline([("prep", prep), ("clf", DecisionTreeClassifier(random_state=42))])
arvore.fit(X_treino, y_treino)
print("recall no treino:", round(recall_score(y_treino, arvore.predict(X_treino)), 2))
print("recall no teste: ", round(recall_score(y_teste, arvore.predict(X_teste)), 2))

from sklearn.neighbors import KNeighborsClassifier
knn = Pipeline([( "prep", prep), ("clf", KNeighborsClassifier(n_neighbors= 15))])
knn.fit(X_treino, y_treino)
print("recall do KNN no teste:" , round(recall_score(y_teste, knn.predict(X_teste)),
2))
print("alertas do KNN:" , knn.predict(X_teste). sum(), "de", len(y_teste))