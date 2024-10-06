import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import euclidean
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

sensus = {
    "tinggi": [158, 170, 183, 191, 155, 163, 180, 158, 178],
    "berat": [64, 86, 84, 80, 49, 59, 67, 54, 67],
    "jk": [
        'pria', 'pria', 'pria', 'pria', 'wanita', 'wanita', 'wanita', 'wanita',
        'wanita'
    ]
}

sensus_df = pd.DataFrame.from_dict(sensus)
print(sensus_df)

fig, ax = plt.subplots()
for jk, d in sensus_df.groupby('jk'):
    ax.scatter(d["tinggi"], d["berat"], label=jk)
plt.legend(loc='upper left')
plt.xlabel("Tinggi")
plt.ylabel("Berat")
plt.grid(True)
plt.show()

X_train = np.array(sensus_df[['tinggi', 'berat']])
y_train = np.array(sensus_df['jk'])

lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)

K = 3
model = KNeighborsClassifier(n_neighbors=K)
model.fit(X_train, y_train)

tinggi_badan = 155
berat_badan = 70
X_new = np.array([tinggi_badan, berat_badan]).reshape(1, -1)

y_new = model.predict(X_new)

print(lb.inverse_transform(y_new))

fig, ax = plt.subplots()
for kl , d in sensus_df.groupby('jk'):
    ax.scatter(d["tinggi"], d["berat"], label=kl)

plt.scatter(tinggi_badan, berat_badan,marker='s',color='red',label="misterius")
plt.legend(loc='upper left')
plt.xlabel("Tinggi")
plt.ylabel("Berat")
plt.grid(True)
plt.show()

misterius = np.array([tinggi_badan, berat_badan])

data_jarak = [euclidean(misterius,d) for d in X_train]
print(data_jarak)

sensus_df['jarak'] = data_jarak
sensus_df.sort_values(['jarak'])

X_test = np.array([[168, 65], [180, 96], [160, 52], [169, 67]])
y_test = lb.transform(np.array(['pria', 'pria', 'wanita', 'wanita'])).flatten()

y_pred = model.predict(X_test)

acc =accuracy_score(y_test,y_pred)
print(acc)

prec = precision_score(y_test,y_pred)
print(f'precision: {prec}')

rec = recall_score(y_test,y_pred)
print(f'recall: {rec}')

f1 = f1_score(y_test,y_pred)
print(f'f1: {f1}')

f1 = 2 * (prec * rec) / (prec + rec)
print(f'f1 score: {f1}')

classification_report = classification_report(y_test,y_pred)
print(classification_report)