import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import pickle
import matplotlib.pyplot as pyplot
from matplotlib import style


data = pd.read_csv("winequality-white.csv", sep=";")

data = data[["fixed acidity", "citric acid", "residual sugar", "chlorides", "total sulfur dioxide", "density", "pH", "alcohol", "quality"]]

predict = "quality" 

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.05)

best = 0

for _ in range(22):
  x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

  linear = linear_model.LinearRegression()

  linear.fit(x_train, y_train)
  acc = linear.score(x_test, y_test)
  print(acc)
  if acc > best:
    best = acc
    with open("city.pickle", "wb") as f:
      pickle.dump(linear, f)


linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)


print("The Best Pct: ", best)
print(acc)
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)

"""for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

#another label
p = 'Elec'
style.use("ggplot")
pyplot.scatter(data[p], data["Elec"])
pyplot.xlabel(p)
pyplot.ylabel("Final grade")
pyplot.show()"""
