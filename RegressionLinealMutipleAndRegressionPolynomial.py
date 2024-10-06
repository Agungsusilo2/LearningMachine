import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt


pizza = {'diameter': [6, 8, 10, 14, 18],
         'n_topping': [2, 1, 0, 2, 0],
         'harga': [7, 9, 13, 17.5, 18]}

train_pizza_df = pd.DataFrame(pizza)
print(train_pizza_df)

pizza = {'diameter': [8, 9, 11, 16, 12],
         'n_topping': [2, 0, 2, 2, 0],
         'harga': [11, 8.5, 15, 18, 11]}

test_pizza_df = pd.DataFrame(pizza)
print(test_pizza_df)

X_train = np.array(train_pizza_df[['diameter','n_topping']])
y_train = np.array(train_pizza_df['harga'])

X_test = test_pizza_df[['diameter','n_topping']]
y_test = test_pizza_df['harga']

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f'r_squared = {r2_score(y_test, y_pred)}')

X_train = np.array(train_pizza_df['diameter']).reshape(-1, 1)
y_train = np.array(train_pizza_df['harga'])

quadratic_feature = PolynomialFeatures(degree=2)
X_train_quadratic = quadratic_feature.fit_transform(X_train)

print(f'X_train_quadratic:\n{X_train_quadratic}\n')

model = LinearRegression()
model.fit(X_train_quadratic, y_train)

X_vis = np.linspace(0,25,100).reshape(-1,1)
X_vis_quadratic = quadratic_feature.transform(X_vis)
y_vis_quadratic = model.predict(X_vis_quadratic)

plt.scatter(X_train,y_train,color='blue')
plt.plot(X_vis, y_vis_quadratic,color='red')

plt.xlim(0,25)
plt.ylim(0,25)
plt.xlabel('Diameter')
plt.ylabel('Harga')
plt.title('Polynomial Regression')
plt.show()

plt.scatter(X_train,y_train,color='blue')
model = LinearRegression()
model.fit(X_train, y_train)
X_vis = np.linspace(0,25,100).reshape(-1,1)
y_vis = model.predict(X_vis)
plt.plot(X_vis, y_vis,color='red')

quadratic_feature = PolynomialFeatures(degree=2)
X_train_quadratic = quadratic_feature.fit_transform(X_train)
model = LinearRegression()
model.fit(X_train_quadratic, y_train)
X_vis_quadratic = quadratic_feature.transform(X_vis)
y_vis_quadratic = model.predict(X_vis_quadratic)
plt.plot(X_vis, y_vis_quadratic,color='red')

cubic_feature = PolynomialFeatures(degree=3)
X_train_quadratic = cubic_feature.fit_transform(X_train)
model = LinearRegression()
model.fit(X_train_quadratic, y_train)
X_vis_quadratic = cubic_feature.transform(X_vis)
y_vis_quadratic = model.predict(X_vis_quadratic)
plt.plot(X_vis, y_vis_quadratic,color='red')

plt.xlabel('Diameter')
plt.ylabel('Harga')
plt.title('Polynomial Regression')
plt.show()
