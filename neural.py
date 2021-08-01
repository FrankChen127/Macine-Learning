import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os


def train_model(train_image, train_labels, times):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(train_image, train_labels, epochs=times)
    model.save("neural_model")
    return model


def load_model():
    model = keras.models.load_model("neural_model")
    return model


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignore warnings from compiling tensorflow

data = keras.datasets.fashion_mnist

(train_image, train_labels), (test_image, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_image = train_image / 255.0  # to shrink the data to 0 ~ 1
test_image = test_image / 255.0
# print(train_image[1])

plt.imshow(train_image[0], cmap=plt.cm.binary)
# plt.show()
'''
上方：資料前處理
---------------------------------------------------------------------------
'''

model = train_model(train_image, train_labels, 100)

test_loss, test_acc = model.evaluate(test_image, test_labels)
print("Loss: ", test_loss, "Accuracy: ", test_acc)

im_predict = test_image[50]

expanded_image = np.expand_dims(im_predict, 0)
predictions = model.predict(expanded_image)

# print(class_names[np.argmax(predictions[0])])
print(class_names[np.argmax(predictions)])
plt.imshow(im_predict, cmap=plt.cm.gray)
plt.title(class_names[np.argmax(predictions)])
#plt.show()

'''
for i in range(5):
    plt.grid(False)
    plt.imshow(test_image[i], cmap=plt.cm.binary)
    plt.xlabel("Correct: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(predictions[i])])
    plt.show()
'''
