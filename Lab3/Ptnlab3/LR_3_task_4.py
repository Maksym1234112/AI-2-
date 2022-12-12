import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, datasets
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()
X = diabetes.data[:, np.newaxis, 2]
Y = diabetes.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=0)
regressor = linear_model.LinearRegression()
regressor.fit(X_train, Y_train)
Y_pred = regressor.predict(X_test)

print(f"Mean squared error = {round(mean_squared_error(Y_test, Y_pred), 2)}")
print(f"Mean absolute error = {round(mean_absolute_error(Y_test, Y_pred), 2)}")
print(f"R2 score = {round(r2_score(Y_test, Y_pred), 2)}")
print(f"Regression coefficient = {round(regressor.coef_[0], 2)}")
print(f"Regression intercept = {round(regressor.intercept_, 2)}")

fig, ax = plt.subplots()
ax.scatter(Y_test, Y_pred, edgecolors=(0, 0, 0))
ax.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
