import os
import glob
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('/NIH_CVP')
df.head()

df.image_path=df.image_path.apply(lambda _str:_str[3:])
df.image_path=df.image_path.apply(lambda _str:_str.replace("\\", "/"))
df.head()

labels = 0, 1
sizes = df.label_value.value_counts().values
explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90);
ax1.legend()

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import shuffle
import keras.backend as K
from functools import partial

num_postive=len(df[df.label_value==1])
num_negative=len(df)-num_postive
sample_ratio=0.3
sample_positive=df[df.label_value==1].iloc[:int(num_postive*sample_ratio)]
sample_negative=df[df.label_value==0].iloc[:int(num_negative*sample_ratio)]
df=pd.concat([sample_positive,sample_negative])
# df=df.sample(frac=1).reset_index(drop=True)

from progressbar import ProgressBar

split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
for train_indices, test_indices in split.split(df, df.label_value):
    train = df.iloc[train_indices]
    test = df.iloc[test_indices]


def load_image(image_path, resize=None):
  global select_method
  img = cv2.imread(image_path)
  if resize is not None:
      img = cv2.resize(img, resize)
  return img / 255


pipeline = partial(load_image, resize=(380, 380))

# training set
train_x = [pipeline(image_path) for image_path in train.image_path.values]
train_x = np.asarray(train_x)
train_y = train.label_value.values.astype(int)
# test set
test_x = [pipeline(image_path) for image_path in test.image_path.values]
test_x = np.asarray(test_x)
test_y = test.label_value.values.astype(int)

plt.imshow(train_x[0])

from efficientnet.keras import EfficientNetB4
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.optimizers import Adam
from keras.regularizers import l1_l2
from keras.models import Model
from keras.applications import *


class CNNBuilder:
  @staticmethod
  # epoch, batch_size, learning_rate, l1_l2, optimizer
  def build(model, input_shape, l1, l2, lr):
      base_model = model(weights='imagenet', include_top=False, input_shape=input_shape)

      x = base_model.output
      x = GlobalAveragePooling2D()(x)
      x = Dropout(0.5)(x)
      output_layer = Dense(1, activation='sigmoid', kernel_regularizer=l1_l2(l1,l2))(x)

      model = Model(inputs=base_model.input, outputs=output_layer)
      model.compile(optimizer='sgd', loss = 'binary_crossentropy',metrics=['accuracy', precision, recall, f1])
      return model

model_mapping = {'DenseNet121':DenseNet121,
 'DenseNet169':DenseNet169,
 'DenseNet201':DenseNet201,
 'InceptionResNetV2':InceptionResNetV2,
 'InceptionV3':InceptionV3,
 'MobileNet':MobileNet,
 'MobileNetV2':MobileNetV2,
 'NASNetLarge':NASNetLarge,
 'NASNetMobile':NASNetMobile,
 'ResNet101':ResNet101,
 'ResNet101V2':ResNet101V2,
 'ResNet152':ResNet152,
 'ResNet152V2':ResNet152V2,
 'ResNet50':ResNet50,
 'ResNet50V2':ResNet50V2,
 'VGG16':VGG16,
 'VGG19':VGG19,
 'EfficientNetB4':EfficientNetB4}

import ipywidgets as widgets


def select_model(change):
  global model_selected
  global model_mapping
  model_selected = model_mapping.get(change.new)


model_selected = VGG16
model_selection=widgets.Select(
    options = model_mapping.keys(),
    value ='VGG16',
    rows = len(model_mapping.keys()),
    description = 'Model',
    disabled = False
)

model_selection.observe(select_model, 'value')

from keras import backend as K


def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall


def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision


def f1(y_true, y_pred):
    _precision = precision(y_true, y_pred)
    _recall = recall(y_true, y_pred)
    return 2*((_precision*_recall)/(_precision+_recall+K.epsilon()))


from keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(
    monitor="loss",
    patience=10,
    verbose=1,
    mode="min",
    restore_best_weights=True,
)


model = CNNBuilder.build(model_selected, (380, 380, 3), 0.005, 0.005, 0.001)
weights = {0:1, 1:2.86}
H = model.fit(train_x, train_y, class_weight=weights,
          steps_per_epoch=len(train_x) / 12, epochs=25, callbacks=[es])

_,axes=plt.subplots(1,2,figsize=(15,7))
# list all data in history
print(H.history.keys())
# summarize history for accuracy
axes[0].plot(H.history['accuracy']);
axes[0].set_title('model accuracy');
axes[0].set_ylabel('accuracy');
axes[0].set_xlabel('epoch');
# summarize history for loss
axes[1].plot(H.history['loss']);
axes[1].set_title('model loss');
axes[1].set_ylabel('loss');
axes[1].set_xlabel('epoch');

#test
import requests
url = r'https://i.ibb.co/kKZZtdR/hard-way-2-0-00-00-00.jpg'
resp = requests.get(url, stream=True).raw
image = np.asarray(bytearray(resp.read()), dtype="uint8")
image = cv2.imdecode(image, cv2.IMREAD_COLOR)
print(image.shape)
plt.figure(figsize=(10,10))
plt.imshow(image,cmap='gray');

norm_image=image/255
norm_image=np.expand_dims(norm_image,axis=0)
model.predict(norm_image)

from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
y_pred = model.predict(test_x)
y_pred[y_pred >= 0.5] = 1
y_pred[y_pred < 0.5] = 0
y_pred = np.squeeze(y_pred)
print(classification_report(test_y, y_pred))
print(accuracy_score(test_y, y_pred))
print(f1_score(test_y, y_pred))
plt.figure(figsize=(10, 10))
sns.heatmap(confusion_matrix(test_y, y_pred), annot=True, square=True)
errors = test_x[np.where((test_y != y_pred) & (test_y == 1))]
for error in errors:
  plt.figure(figsize=(10, 16))
  plt.imshow(error)
  plt.show()

errors = test_x[np.where((test_y != y_pred) & (test_y == 0))]
for error in errors:
  plt.figure(figsize=(10, 16))
  plt.imshow(error)
  plt.show()













