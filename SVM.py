# svm stands for Support Vector Machine
import sklearn.model_selection
from sklearn import svm
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot

breast = datasets.load_breast_cancer()
#print(breast.feature_names)
#print(breast.target_names)

X = breast.data
y = breast.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.2)
target = ['malignant' 'benign']

model = svm.SVC(kernel="linear", C = 2)
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
acc = metrics.accuracy_score(y_predict, y_test)
print(acc)
for i in range(len(y_test)):
    print("predict: ",y_predict[i], "real: ", y_test[i])