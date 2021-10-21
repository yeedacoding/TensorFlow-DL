from dataset import test_data, test_labels
from dataexplore import train_data, train_labels
from model_build import history

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

# model.fit()는 history객체를 반환
# history에는 훈련하는 동안 일어난 모든 정보가 담긴 딕셔너리가 들어있음

history_dict = history.history
history_dict.keys()

# 훈련손실, 검증손실 그래프와 훈련 정확도, 검증정확도 그래프를 그려서 비교하기
# 점선 = 훈련손실, 훈련정확도
# 실선 = 검증손실, 검증정확도


acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# 'bo' = '파란색 점
# 'b' = '파란 실선'

# 훈련손실, 검증손실 그래프
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# 훈련정확도, 검증정확도
plt.clf()  # 그림 초기화

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
