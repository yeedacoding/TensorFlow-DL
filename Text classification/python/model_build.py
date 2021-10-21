from dataset import test_data, test_labels
from dataexplore import train_data, train_labels

import tensorflow as tf
from tensorflow import keras

import numpy as np

#################################################################
# 모델 구성
#################################################################

# 역시 신경망은 layer를 쌓아서 만드네
# 고려사항
# 1. 모델에서 얼마나 많은 층을 사용할 것인가?
# 2. 각 층에서 얼마나 많은 은닉 유닛(hidden unit)을 사용할 것인가?
# 이 예제의 입력 데이터 = 단어 인덱스의 배열
# 레이블 = 0 또는 1

# 입력 크기는 영화 리뷰 데이터셋에 적용된 어휘 사전의 크기(10000개의 단어)
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16, input_shape=(None, )))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()

# 모델에 사용할 optimizer 와 loss function 설정

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 검증 세트 만들기
# 기존 train_data에서 10000개의 샘플을 떼어내어 검증 세트(validation set)를 만들기
# test_data를 사용하지 않는 이유 = train_data만을 사용하여 모델을 개발하고 튜닝하는 것이 목표

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

#################################################################
# 모델 훈련
#################################################################

# 512개의 샘플로 이루어진 미니배치에서 40번의 에포크 동안 훈련
# x_train과 y_train 텐서에 있는 모든 샘플에 대해 40번 반복한다는 뜻

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

#################################################################
# 모델 평가
#################################################################

# 모델 성능 확인하기
# 손실값과 정확도 반환 (손실=오차이므로 숫자가 낮을수록 좋음)

results = model.evaluate(test_data, test_labels, verbose=2)

print(results)
