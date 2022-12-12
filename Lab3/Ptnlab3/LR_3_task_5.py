import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

m = 100
X = 6 * np.random.rand(m, 1) - 5
Y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

indices = np.argsort(X, axis=0)
X = X[indices].reshape(-1, 1)
Y = Y[indices].reshape(-1, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=0)
regressor = linear_model.LinearRegression()
regressor.fit(X_train, Y_train)
Y_pred = regressor.predict(X_test)

print(f"Mean absolute error = {round(mean_absolute_error(Y_test, Y_pred), 2)}")
print(f"Mean squared error = {round(mean_squared_error(Y_test, Y_pred), 2)}")
print(f"Regression coefficient = {round(regressor.coef_[0][0], 2)}")
print(f"Regression intercept = {round(regressor.intercept_[0], 2)}")
print(f"R2 score = {round(r2_score(Y_test, Y_pred), 2)}")

plt.scatter(X, Y, edgecolors=(0, 0, 0))
plt.plot(X_test, Y_pred, color="red")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
regressor = linear_model.LinearRegression()
regressor.fit(X_poly, Y)
Y_pred = regressor.predict(X_poly)

print(f"Mean absolute error = {round(mean_absolute_error(Y, Y_pred), 2)}")
print(f"Mean squared error = {round(mean_squared_error(Y, Y_pred), 2)}")
print(f"Regression coefficient = {round(regressor.coef_[0][0], 2)}")
print(f"Regression intercept = {round(regressor.intercept_[0], 2)}")
print(f"R2 score = {round(r2_score(Y, Y_pred), 2)}")

plt.scatter(X, Y, edgecolors=(0, 0, 0))
plt.plot(X, Y_pred, color="red")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
