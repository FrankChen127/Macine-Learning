import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
import sklearn
warnings.simplefilter("ignore", category = FutureWarning)

data = pd.read_csv("student-mat.csv", sep = ";")
data = data[["G1", "G2", "G3", "absences", "studytime", "freetime", "Walc"]]

X = np.array(data.drop(["G3"], 1))     # 1 means the axis of the "G3", in this case it is one(2D list)
Y = np.array(data["G3"])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state= 20)

pickle_in = open("student_model.pickle", "rb")
linear = pickle.load(pickle_in)
print(type(linear))

acc = linear.score(X_test, Y_test)
print(acc)
prediction = linear.predict(X_test)
print(prediction[0])
