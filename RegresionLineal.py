import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

df = {'diameter': [6, 8, 10, 14, 18],
      'harga': [7, 9, 13, 17.5, 18]}

pizza_df = pd.DataFrame(df)
print(pizza_df)

pizza_df.plot(x='diameter', y='harga', kind='scatter')
plt.xlabel('Diameter')
plt.ylabel('Harga')
plt.xlim(0, 25)
plt.ylim(0, 25)
plt.show()

X = np.array(pizza_df['diameter'])
Y = np.array(pizza_df['harga'])

X = X.reshape(-1, 1)
model = LinearRegression()
model.fit(X, Y)

x_vis = np.array([0, 25]).reshape(-1, 1)
y_vis = model.predict(x_vis)

plt.scatter(X, Y)
plt.plot(x_vis, y_vis)
plt.show()

print(model.intercept_)
print(model.coef_)

var_x = np.var(X.flatten(),ddof=1)
print(var_x)

cov_xy = np.cov(X.flatten(),Y)[0][1]

slope = cov_xy / var_x

intercept = np.mean(Y) - slope * np.mean(X)
print(intercept)

diameter_pizza = np.array([10,14,15,20]).reshape(-1, 1)
prediction = model.predict(diameter_pizza)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.4,
                                                    random_state=1)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

ss_res = sum([(y_i - model.predict(x_i.reshape(-1, 1))[0])**2
              for x_i, y_i in zip(X_test, y_test)])

print(f'ss_res: {ss_res}')

mean_y = np.mean(y_test)
ss_tot = sum([(y_i - mean_y)**2 for y_i in y_test])

print(f'ss_tot: {ss_tot}')

r_squared = 1 - (ss_res / ss_tot)
print(f'r_squared: {r_squared}')