import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

transacoes = pd.read_csv("transacoes.csv")
X = transacoes.drop(columns=["id_transacao", "fraude"])
y = transacoes["fraude"]
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)
print(X_treino.shape, X_teste.shape)
print("fraudes no treino:", y_treino.sum(), "| no teste:", y_teste.sum())

for semente in [1, 2, 3]:
    _, _, _, y_t = train_test_split(X, y, test_size=0.25, random_state=semente)
    print("semente", semente, "-> fraudes no teste sem stratify:", y_t.sum())

base = DummyClassifier(strategy="most_frequent")
base.fit(X_treino, y_treino)
print("acurácia:", round(base.score(X_teste, y_teste), 3))
print("previsões distintas:", np.unique(base.predict(X_teste)))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_teste, base.predict(X_teste))
print(cm)
vn, fp, fn, vp = cm.ravel()
print("VN", vn, "FP", fp, "FN", fn, "VP", vp)


from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(
 y_teste, base.predict(X_teste),
display_labels=["Legítima", "Fraude"], colorbar=False)
plt.grid(False)
plt.show()


base = DummyClassifier(strategy="most_frequent").fit(X_treino, y_treino)
y_prev = base.predict(X_teste)

vn, fp, fn, vp = confusion_matrix(y_teste, y_prev).ravel()
print("VN", vn, "FP", fp, "FN", fn, "VP", vp)
print("precision:", precision_score(y_teste, y_prev, zero_division=0))
print("recall:   ", recall_score(y_teste, y_prev))
print("f1:       ", f1_score(y_teste, y_prev))
print(classification_report(y_teste, y_prev, target_names=["Legítima", "Fraude"], zero_division=0))
print(f"prejuízo: R$ {fn * 800 + fp * 15:,.0f}".replace(",", "."))