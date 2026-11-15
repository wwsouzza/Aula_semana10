import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


#Preparação dos dados para teste
transacoes = pd.read_csv( "transacoes.csv" )
X = transacoes.drop(columns=[ "id_transacao" , "fraude"])
y = transacoes[ "fraude"]
X_treino, X_teste, y_treino, y_teste = train_test_split(
 X, y, test_size= 0.25, stratify=y, random_state= 42)
print(y_treino.value_counts().to_dict())

# Pipeline de classificação
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
numericas = [ "valor", "hora", "distancia_residencia_km" , "tentativas_24h" ,
"idade_conta_dias" ]
prep = ColumnTransformer([( "num", StandardScaler(), numericas),
 ( "cat", OneHotEncoder(), [ "canal"])], remainder= "passthrough" )
print("colunas na saída:" , prep.fit_transform(X_treino).shape[ 1])


#peso para as classes
from sklearn.utils.class_weight import compute_class_weight
pesos = compute_class_weight( "balanced" , classes=np.array([ 0, 1]), y=y_treino)
print(pesos.round(2))
n, k = len(y_treino), 2
print("manual:", round(n / (k * (y_treino == 0).sum()), 2), round(n / (k * (y_treino == 1).sum()),
2))


# Função resumo introduzida (métricas)
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from sklearn.tree import DecisionTreeClassifier
def resumo(nome, modelo, X_teste, y_teste):
 prev = modelo.predict(X_teste)
 proba = modelo.predict_proba(X_teste)[:, 1]
 return {"técnica": nome,
 "precision": precision_score(y_teste, prev, zero_division=0),
 "recall": recall_score(y_teste, prev),
 "f1": f1_score(y_teste, prev),
 "pr_auc": average_precision_score(y_teste, proba)}
print("função resumo pronta")

# regressão logística sem peso, com class_weight='balanced' e manual
resultados = []
for nome, cw in [("sem técnica", None), ("class_weight", "balanced"), ("peso 1:10", {0: 1, 1: 10})]:
 m = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=1000, class_weight=cw))])
 m.fit(X_treino, y_treino)
 resultados.append(resumo(nome, m, X_teste, y_teste))
print(pd.DataFrame(resultados).round(3).to_string(index=False))


