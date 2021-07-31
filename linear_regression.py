import timeit
start = timeit.default_timer()
import tensorflow
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import warnings
import pickle
from matplotlib import style
from matplotlib import pyplot as plt

stop = timeit.default_timer()
print(stop - start)
start = timeit.default_timer()
warnings.simplefilter(action="ignore", category=FutureWarning)  # ignore FutureWarning(pandas)

data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "freetime", "absences", "Dalc", "health", "goout"]]  # choose the data I want
print(data)

predict = "G3"
X = np.array(data.drop([predict], 1))  # data without predict
y = np.array(data[predict])            # data with only predict

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)  # only used on large data set

'''
best = 0
print("Running......")
for i in range(10000):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
    linear = linear_model.LinearRegression()
    linear.fit(X_train, y_train)
    acc = linear.score(X_test, y_test)

    if best < acc:
        with open("student_model2.pickle", "wb") as file: #save linear to student_model.pickle
            pickle.dump(linear, file)
        best = acc
'''

pickle_in = open("student_model.pickle", "rb")
linear = pickle.load(pickle_in)
acc = linear.score(X_test, y_test)

print("Coefficient", linear.coef_)       # 係數
print("Intercepts ", linear.intercept_)  # 截距

predictions = linear.predict(X_test)
mse = sklearn.metrics.mean_squared_error(y_test, predictions)
print(mse)

for i in range(len(predictions)):
    print(predictions[i], X_test[i], y_test[i])
print(len(X_test))

p = 'absences'

style.use("seaborn-poster")
plt.scatter(data[p], data[predict])
plt.xlabel(p)
plt.ylabel(predict)
plt.show()


stop = timeit.default_timer()
print("Time: ", stop - start)
print(acc)


