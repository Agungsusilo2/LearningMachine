import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer,StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.spatial.distance import euclidean

sensus = {
    'tinggi': [158, 170, 183, 191, 155, 163, 180, 158, 170],
    'jk': ['pria', 'pria', 'pria', 'pria', 'wanita', 'wanita', 'wanita', 'wanita', 'wanita'],
    'berat': [64, 86, 84, 80, 49, 59, 67, 54, 67]
}

sensus_df = pd.DataFrame(sensus)
print(sensus_df)

X_train = sensus_df[['tinggi', 'jk']]
y_train = sensus_df['berat']

# Binarize the 'jk' column
lb = LabelBinarizer()
X_train['jk'] = lb.fit_transform(X_train['jk'])

# Convert to numpy array

X_train = X_train.values

K = 3
model = KNeighborsRegressor(n_neighbors=K)
model.fit(X_train, y_train)

X_test = np.array([[168, 0], [180, 0], [160, 1], [169, 1]])
y_test = np.array([65, 96, 52, 67])

y_pred = model.predict(X_test)
r_squared = r2_score(y_test, y_pred)
print(f'R-squared: {r_squared}')

mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

MSE = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {MSE}')

X_train = np.array([[1700, 0], [1600, 1]])
X_new = np.array([[1640, 0]])

# print([euclidean(X_new[0],d) for d in X_train])

ss = StandardScaler()
X_train = np.array([[1700,0], [1600,1]])
X_train_scaled = ss.fit_transform(X_train)

X_new = np.array([[1640,0]])
x_new_scaled = ss.transform(X_new)

jarak = [euclidean(x_new_scaled[0],d) for d in X_train_scaled]

X_train = np.array([[1.7, 0], [1.6, 1]])
X_train_scaled = ss.fit_transform(X_train)
print(f'X_train_scaled:\n{X_train_scaled}\n')

X_new = np.array([[1.64,0]])
x_new_scaled = ss.transform(X_new)
print(f'x_new_scaled:\n{x_new_scaled}\n')

jarak = [euclidean(x_new_scaled[0],d) for d in X_train_scaled]

# Training Set
X_train = np.array([[158, 0], [170, 0], [183, 0], [191, 0], [155, 1], [163, 1],
                    [180, 1], [158, 1], [170, 1]])

y_train = np.array([64, 86, 84, 80, 49, 59, 67, 54, 67])

# Test Set
X_test = np.array([[168, 0], [180, 0], [160, 1], [169, 1]])
y_test = np.array([65, 96, 52, 67])

X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)
print(f'X_train_scaled:\n{X_train_scaled}\n')
print(f'X_new_scaled:\n{X_test_scaled}\n')

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')
MSE = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {MSE}')