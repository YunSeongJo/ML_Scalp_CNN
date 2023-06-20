from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
import os
from keras.preprocessing.image import ImageDataGenerator
import warnings

import matplotlib.pyplot as plot
warnings.filterwarnings("ignore", category=UserWarning)
import tensorflow as tf
from tensorflow.python.client import device_lib

# gpu 설정 및 할당
print(device_lib.list_local_devices())
print(tf.__version__)
print(tf.test.is_gpu_available())

tf.config.list_logical_devices()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 데이터 불러오기 정상: 2136개, 비듬: 2256개, 모낭홍반농포: 1328개, 탈모: 2508개


# 랜덤 시드 고정시키기
np.random.seed(3)

# 전체 훈련데이터 8228개를 8:2로 나누어서 검증데이터 생성

# 훈련 데이터 6592개를 불러와서 전처리
train_datagen = ImageDataGenerator(rescale= 1./255)
train_generator = train_datagen.flow_from_directory(
    './TrainData',
    target_size = (128, 128),
    batch_size = 64,
    class_mode = 'categorical'
)
# 검증 데이터 1636개를 전처리
test_datagen = ImageDataGenerator(rescale = 1./255)
test_generator = test_datagen.flow_from_directory(
    './ValidData',
    target_size = (128, 128),
    batch_size = 64,
    class_mode = 'categorical'
)
# 테스트 데이터 40개 전처리
test_datagen = ImageDataGenerator(rescale = 1./255)
test_generator2 = test_datagen.flow_from_directory(
    './TestData',
    target_size = (128, 128),
    batch_size = 1,
    class_mode = 'categorical'
)



# 모델 구성하기
model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3),
                 activation = 'relu',
                 input_shape = (128, 128, 3)))
model.add(Conv2D(64, (3, 3),
                 activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(4, activation = 'softmax'))

# 모델 학습 과정
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

#모델 학습
data = model.fit_generator(train_generator,
                    steps_per_epoch = 103,
                    epochs = 10,
                    validation_data = test_generator,
                    validation_steps = 26)

# 학습된 그래프 보기
plot.plot(data.history['accuracy'])
plot.plot(data.history['val_accuracy'])
plot.title('Model accuracy')
plot.ylabel('Accuracy')
plot.xlabel('Epoch')
plot.legend(['Train', 'Test'], loc='upper left')
plot.show()

plot.plot(data.history['loss'])
plot.plot(data.history['val_loss'])
plot.title('Model loss')
plot.ylabel('Loss')
plot.xlabel('Epoch')
plot.legend(['Train', 'Test'], loc='upper left')
plot.show()

# 모델 평가
print("--evaluate")
scores = model.evaluate_generator(test_generator, steps = 26)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))


# 모델 사용
print("--Predict--")
output = model.predict_generator(test_generator2, steps = 40)
np.set_printoptions(formatter = {'float' : lambda x : "{0:0.3f}".format(x)})
print(test_generator2.class_indices)
#print(output)


image_names = test_generator2.filenames

#예측 결과와 이미지 이름 출력
for i in range(len(output)):
    print("Image:", image_names[i])
    print("Prediction:", output[i])


# pred_class = []
#
# for i in output:
#     pred = np.argmax(i)
#     pred_class.append(pred)
#
# print(pred_class)

