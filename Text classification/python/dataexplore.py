from dataset import train_data, train_labels, test_data, test_labels
from dataset import word_index

import tensorflow as tf
from tensorflow import keras

import numpy as np

# 리뷰(정수 배열)는 신경망에 주입하기 전에 텐서로 변환되어야 함
# 1. 원-핫 인코딩(one-hot encoding) = 정수 배열을 0과 1로 이루어진 벡터로 변환
# 2. 정수 배열의 길이가 모두 같도록 패딩(padding)을 추가해 max_length * num_reviews 크기의 정수 텐서를 만듦

# 모든 영화 리뷰의 길이가 같아야 하므로 pad_sequences함수를 사용해 길이 맞추기

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

print(len(train_data[0]), len(train_data[1]))
print(train_data[0])
# 빈 배열은 0이 되는구나
