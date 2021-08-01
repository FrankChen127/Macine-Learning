from tensorflow import keras
import numpy as np
import tensorflow

data = keras.datasets.imdb

(train_data, train_label), (test_data, test_label) = data.load_data(num_words=1000)
word_index = data.get_word_index()

word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
print(len(train_data))

for i in range(len(test_label), 0):
    if test_label[i] == 1:
        value = i
        break


def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


test_data = keras.preprocessing.sequence.pad_sequences(test_data, value = word_index["<PAD>"], maxlen = 250, padding="post")
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value = word_index["<PAD>"], maxlen = 250, padding="post")
#print(decode_review(x_test[0]))
#print(len(x_train[0]))

'''
# model down there
model = keras.models.Sequential()
model.add(keras.layers.Embedding(88000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))
#model.summary()

model.compile(optimizer = "adam", loss ="binary_crossentropy", metrics=["accuracy"])

val = 9000
x_val = train_data[:val]
x_train = train_data[val:]

y_val = train_label[:val]
y_train = train_label[val:]

model.fit(x_train, y_train, epochs=50, batch_size=600, validation_data=(x_val, y_val), verbose=1)
'''
model = keras.models.load_model("text_model.h5")
result = model.evaluate(test_data, test_label)
print(result)
#model.save("text_model.h5")

test_review = test_data[221]
predict = model.predict([test_review])
print("Review: ")
print(decode_review(test_review))
print("Prediction: " + str(predict[0]))
print("Actual: " + str(test_label[221]))