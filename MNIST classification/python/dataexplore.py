from dataset import train_images, train_labels, test_images, test_labels
from dataset import class_names

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 데이터 전처리

train_images = train_images / 255.0
test_images = test_images / 255.0


plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
