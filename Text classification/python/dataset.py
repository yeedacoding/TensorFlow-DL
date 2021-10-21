import tensorflow as tf
from tensorflow import keras

import numpy as np

# IMDB 데이터셋 다운로드

imdb = keras.datasets.imdb

(train_data, train_labels), (test_data,
                             test_labels) = imdb.load_data(num_words=10000)
# num_words = 10000 은 훈련 데이터에서 가장 많이 등장하는 상위 10000개의 단어

# 데이터 형태 살펴보기
# IMDB 데이터셋의 샘플은 전처리된 정수 배열 -> 정수는 영화 리뷰에 나오는 단어
# 레이블은 정수 0 또는 1 -> 0 = 부정적 리뷰, 1 = 긍정적 리뷰

print("훈련 샘플 : {}, 레이블 : {}".format(len(train_data), len(train_labels)))
print("테스트 샘플 : {}, 레이블 : {}".format(len(test_data), len(test_labels)))

# 리뷰 텍스트는 어휘 사전의 특정 단어를 나타내는 정수로 변환되어 있음
# 첫번째 훈련 데이터에 들어있는 리뷰 텍스트

print(train_data[0])

# 당연히 모든 데이터(리뷰 텍스트)의 길이는 다름
# 신경망의 입력은 길이가 같아야 하기 때문에 이 문제를 해결해야 함

print(len(train_data[0]), len(train_data[1]))

# 정수를 다시 단어로 변환하기
# 정수와 문자열을 매핑한 딕셔너리(dictionary) 객체에 질의하는 헬퍼(helper)함수를 만들어보기

# 단어와 정수 인덱스를 매핑한 딕셔너리
word_index = imdb.get_word_index()

# 처음 몇 개 인덱스는 사전에 정의되어 있음
word_index = {k: (v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<STAR>"] = 1
word_index["<UNK>"] = 2  # UNKOWN
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key)
                          for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


print(decode_review(train_data[0]))
