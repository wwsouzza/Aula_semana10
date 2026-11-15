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

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE

X_prep = prep.fit_transform(X_treino)
rus = RandomUnderSampler(random_state= 42)
X_sub, y_sub = rus.fit_resample(X_prep, y_treino)
print("antes: ", y_treino.value_counts().to_dict())
print("depois:", y_sub.value_counts().to_dict())
print(X_prep.shape, "->", X_sub.shape)

for estrategia in ["auto", 0.1, 0.5, {0: 600, 1: 139}]:
 _, y_novo = RandomUnderSampler(sampling_strategy=estrategia, random_state= 42).fit_resample(X_prep, y_treino)
 print(f"{str(estrategia):18s} -> {y_novo.value_counts().to_dict()}")

ros = RandomOverSampler(random_state= 42)
X_sup, y_sup = ros.fit_resample(X_prep, y_treino)
print("depois:", y_sup.value_counts().to_dict())
print("linhas únicas entre as fraudes:" , np.unique(X_sup[y_sup == 1],
axis=0).shape[0])

smote = SMOTE(random_state=42)
X_sm, y_sm = smote.fit_resample(X_prep, y_treino)
print("depois:", y_sm.value_counts().to_dict())
print("linhas únicas entre as fraudes:", np.unique(X_sm[y_sm == 1], axis=0).shape[0])