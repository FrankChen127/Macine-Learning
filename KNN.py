import timeit
start = timeit.default_timer()
import sklearn
from sklearn.utils import shuffle
import pandas as pd
import matplotlib
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot
from matplotlib.style import use

stop = timeit.default_timer()
#print(stop - start)
import warnings
warnings.simplefilter("ignore", category = FutureWarning)

data = pd.read_csv("car.data", sep = ",")
print(data.head())

encoder = sklearn.preprocessing.LabelEncoder()
buying = list(encoder.fit_transform(data["buying"]))
maint = list(encoder.fit_transform(data["maint"]))
lug_boot = list(encoder.fit_transform(data["lug_boot"]))
safety = list(encoder.fit_transform(data["safety"]))
classes = list(encoder.fit_transform(data["classes"]))
doors = list(encoder.fit_transform(data["doors"]))
persons = list(encoder.fit_transform(data["persons"]))

predict = "classes"

X = list(zip(buying, maint, lug_boot, persons, doors, safety))
y = list(classes)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

'''
matplotlib.style.use("ggplot")
matplotlib.pyplot.scatter(classes, maint)
matplotlib.pyplot.xlabel("classes")
matplotlib.pyplot.ylabel("maint")
matplotlib.pyplot.show()
'''

model = KNeighborsClassifier(n_neighbors = 7)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)
predictions = model.predict(x_test)
name = ["unacc", "acc", 'good', 'vgood']

for i in range(len(y_test)):
    print("Predicted: ", name[predictions[i]], " Actually: ", name[y_test[i]])
    n = model.kneighbors([x_test[i]], 7)
    print("N: ", n)
