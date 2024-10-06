import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.datasets import fetch_openml
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

fig, axes = plt.subplots(1, 15, figsize=(15, 2))

for i, ax in enumerate(axes):
    ax.imshow(X.to_numpy()[i].reshape(28, 28), cmap=cm.get_cmap("Purples_r"))
    ax.axis('off')
    ax.set_title(y[i])

plt.show()

X_train = X[:1000]
y_train = y[:1000]
X_test = X[69000:]
y_test = y[69000:]

model = SVC(random_state=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred)
print(classification_report(y_test, y_pred))

parameters = {'kernel': ['linear', 'poly', 'rbf'], 'C': [1, 10, 100, 1000],'gamma': [0.001, 0.01, 0.1, 1, 10]}
grid_search = GridSearchCV(estimator=SVC(random_state=0), param_grid=parameters, scoring='accuracy',verbose=True)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
best_params = grid_search.best_estimator_.get_params()
for param in parameters:
    print(param, grid_search.best_score_, grid_search.best_params_)

y_pred = grid_search.predict(X_test)
print(classification_report(y_test, y_pred))
