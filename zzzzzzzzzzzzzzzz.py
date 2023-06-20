import PIL
import numpy as np
# np.set_printoptions(threshold=np.inf, linewidth=np.inf)
import os
from glob import glob
from PIL import Image
import matplotlib.pyplot as plot
from keras.preprocessing.image import ImageDataGenerator
import warnings
import tensorflow as tf
from tensorflow.python.client import device_lib

from matplotlib import pyplot as plt


warnings.filterwarnings("ignore", category=UserWarning)


# gpu 설정 및 할당
print(device_lib.list_local_devices())
print(tf.__version__)
print(tf.test.is_gpu_available())

tf.config.list_logical_devices()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 데이터 불러오기 정상: 2136개, 비듬: 2256개, 모낭홍반농포: 1328개, 탈모: 2508개 // Test: 1113개
train_path1 = "./Final_input_data/"
test_path = "./AAAAAAAAAAAAAAAAA/"

# train 이미지를 불러오고, label은 image가 저장된 폴더의 이름으로 설정하여 총 8228개를 가져온다.
training_images = []
training_labels = []
testing_images = []

for filename in glob(train_path1 + "*"):
    for img in glob(filename + "/*.jpg"):
        an_img = PIL.Image.open(img).resize((int(128), int(128)))  # read img
        img_array = np.array(an_img)  # img to array
        training_images.append(img_array)  # append array to training_images
        label = filename.split('\\')[-1]  # get label
        training_labels.append(label)  # append label

training_images = np.array(training_images)
training_labels = np.array(training_labels)


# for filename in glob(test_path + "*"):
#     for img in glob(filename + "/*.jpg"):
#         an_img = PIL.Image.open(img).resize((int(32), int(32)))  # read img & resize
#         img_array = np.array(an_img)  # img to array
#         training_images.append(img_array)  # append array to training_images
#         label = filename.split('\\')[-1]  # get label
#         training_labels.append(label)  # append label
#
# testing_images = np.array(testing_images)
from sklearn.preprocessing import LabelEncoder

# LabelEncoder 클래스를 사용하여 범주형 레이블을 숫자로 인코딩하는 과정
le = LabelEncoder()
training_labels = le.fit_transform(training_labels)
training_labels = training_labels.reshape(-1, 1)

print("training_images.shape", training_images.shape)
print("training_labels.shape", training_labels.shape)

# for i in range(20):
#     plt.subplot(4, 5, i + 1)
#     plt.imshow(training_images[i*400])
#     print(training_labels[i*400], end=",")
#
# plt.show()
# test에서 이미지를 가져오지만 label은 우리가 추론해야 하는 값이기 때문에 주어지지 않는다.
test_images = []
test_idx = []

flist = sorted(glob(test_path + '*.jpg'))
for filename in flist:
    an_img = PIL.Image.open(img).resize((int(128), int(128)))  # read img
    img_array = np.array(an_img)  # img to array
    test_images.append(img_array)  # append array to training_images
    label = filename.split('\\')[-1]  # get id
    test_idx.append(label)  # append id
test_images = np.array(test_images)
print("test_images shape:", test_images.shape)
print(test_images, "TLlllllllltoyvaiytaityvaitaiwlytbvawiltbyawvbyawilotybvawilbyvtialwtiwavybwila")
print(test_idx[0:5])

label_num = [0, 1, 2, 3]
label_name = le.inverse_transform(label_num)
for i in range(4):
    print(label_num[i], label_name[i])
#training_labels = tf.one_hot(training_labels, 4)
#training_labels = np.reshape(training_labels,(-1,1)) # 원래(-1,4)였음
print(training_labels)



# 데이터 전처리 과정 -----
# data augmentation을 통한 데이터 생성, keras에서 제공하는 ImageDataGenerator 함수를 사용하여 데이터를 생성
# 데이터를의 개수를 늘려서 overfitting을 해결하기 위함
tf.random.set_seed(42)

image_generator = ImageDataGenerator(
    rotation_range=90, # 이미지를 랜덤한 각도로 돌리는 정도 원래 30이였음
    brightness_range=[0.8, 1.0], # 이미지의 밝기를 랜덤하게 다르게 주는 정도
    zoom_range=0.3, # 사진을 확대하는 정도
    width_shift_range=0.2, # 사진을 왼쪽 오른쪽으로 움직이는 정도
    height_shift_range=0.2, # 사진을 위 아래로 움직이는 정도
    horizontal_flip=True, # y축을 기준으로 반전 (오른쪽 왼쪽 뒤집기)
    vertical_flip=False #  x축을 기준으로 반전 (위 아래 뒤집기) 원래 False였음
)

augment_size = 20000  # 데이터를 추가할 개수, 총 108228개로 증가함

np.random.seed(42)

random_mask = np.random.randint(training_images.shape[0], size=augment_size)
training_image_aug = training_images[random_mask].copy()
training_labels_aug = training_labels[random_mask].copy()

training_image_aug = \
    image_generator.flow(training_image_aug, np.zeros(augment_size), batch_size=augment_size, shuffle=False,
                         seed=42).next()[0]

training_images = np.concatenate((training_images, training_image_aug))
training_labels = np.concatenate((training_labels, training_labels_aug))

print(training_images.shape)
print(training_labels.shape)


# mixup augmentation
# 증가한 이미지 말고 기본 이미지 8228개에서 2개의 이미지를 선택하여 선형 결합해서 새로운 이미지 데이터를 만들어내는 기법
# 그리하여 기존의 8228개 + 믹스한 이미지 4114개 == 12342개
# sample_image = training_images[1]
# sample_label = tf.one_hot(training_labels[1], 4)  # one hot encoding을진행해야 mixup을 할 수 있습니다.
#
# sample_image2 = training_images[5001]
# sample_label2 = tf.one_hot(training_labels[5001], 4)


# fig = plt.figure(figsize=(8, 8))
#
# plt.subplot(2, 2, 1)
# plt.imshow(sample_image.astype('uint8'))
# print(sample_label)
#
# plt.subplot(2, 2, 2)
# plt.imshow(sample_image2.astype('uint8'))
# print(sample_label2)
#
# plt.show()

# def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
#     gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
#     gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
#     return gamma_1_sample / (gamma_1_sample + gamma_2_sample)
#
#
# def mix_up(ds_one, ds_two, batch_size=1, alpha=0.2):
#     # Unpack two datasets
#     images_one, labels_one = ds_one
#     images_two, labels_two = ds_two
#
#     # Sample lambda and reshape it to do the mixup
#     l = sample_beta_distribution(batch_size, alpha, alpha)
#     x_l = tf.reshape(l, (batch_size, 1, 1, 1))
#     y_l = tf.reshape(l, (batch_size, 1))
#
#     # Perform mixup on both images and labels by combining a pair of images/labels
#     # (one from each dataset) into one image/label
#     images = images_one * x_l + images_two * (1 - x_l)
#     labels = labels_one * y_l + labels_two * (1 - y_l)
#     return (images, labels)


# mix_image, mix_label = mix_up((sample_image, sample_label), (sample_image2, sample_label2), batch_size=9, alpha=0.5)

# fig = plt.figure(figsize = (10,10))
# for i in range(9):
#     plt.subplot(3, 3, 1+i)
#     image = mix_image[i]
#     plt.imshow(image.numpy().squeeze().astype('uint8'))
#     plt.show()
#     print(mix_label[i].numpy().tolist())

# import random
#
# random.seed(42)
# from sys import stdout
#
training_labels = tf.one_hot(training_labels, 4)  # mixup을 적용하기 위해 one-hot 기법을 적용해줍니다
print(training_labels)
#
# mix_training_images = []
# mix_training_labels = []
#
# for i in range(3):
#     random_num = random.sample(range(0, 8228), 8228)  # augmentation을 적용한 데이터를 제외하고 mix해보겠습니다
#     print("\nAttempt", i)
#     progress_before = 0
#
#     for i in range(0, 8228, 2):
#         image_1 = training_images[random_num[i]]
#         label_1 = training_labels[random_num[i]]
#
#         image_2 = training_images[random_num[i + 1]]
#         label_2 = training_labels[random_num[i + 1]]
#
#         mix_image, mix_label = mix_up((image_1, label_1), (image_2, label_2))
#
#         mix_training_images.append(mix_image[0])
#         mix_training_labels.append(mix_label[0])
#
#         # just for UI
#         progress = int(100 * (i / 8228))
#         if progress != progress_before:
#             progress_before = progress
#             stdout.write("\r ========= %d%% completed =========" % progress)
#             stdout.flush()
#
# mix_training_images = np.array(mix_training_images)
# mix_training_labels = np.array(mix_training_labels)
#
# print('mix_train 크기:', mix_training_images.shape)
# print('mix_label 크기:', mix_training_labels.shape)


# Data set나누기 과정

from sklearn.model_selection import train_test_split

training_labels = np.array(training_labels)
training_labels = training_labels.reshape(-1, 4)  # mixup에서 one-hot 기법을 적용하여 shape를 바꿈
print(training_labels)

X_train, X_valid, y_train, y_valid = train_test_split(training_images,
                                                      training_labels,
                                                      test_size=0.1,
                                                      stratify=training_labels,
                                                      random_state=42)

# X_train = np.concatenate((X_train, mix_training_images))  # mixup한 4114개의 데이터를 train set에 추가해줍니다
# y_train = np.concatenate((y_train, mix_training_labels))

X_test = test_images

#따라서 12342 + 100000개 == 112342개가 되고, 여기서 5%를 validation set로 나누고 나머지 95%은 train으로 사용
print('X_train 크기:', X_train.shape)
print('y_train 크기:', y_train.shape)
print('X_valid 크기:', X_valid.shape)
print('y_valid 크기:', y_valid.shape)
print('X_test  크기:', X_test.shape)

X_train = X_train / 255.0
X_valid = X_valid / 255.0
X_test = X_test / 255.0

# 모델 생성 과정

model = tf.keras.models.Sequential([
    # tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='SAME', input_shape=(128, 128, 3)),  # cnn layer
    # tf.keras.layers.MaxPooling2D(2, 2),  # batch norm layer
    # tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='SAME'),  # cnn layer
    # tf.keras.layers.MaxPooling2D(2, 2),  # batch norm layer
    # tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='SAME'),  # cnn layer
    # tf.keras.layers.MaxPooling2D(2, 2),  # batch norm layer
    # tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='SAME'),  # cnn layer
    # tf.keras.layers.MaxPooling2D(2, 2),  # batch norm layer
    # tf.keras.layers.Flatten(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='SAME', input_shape=(128, 128, 3)),  # cnn layer
    tf.keras.layers.BatchNormalization(),  # batch norm layer
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='SAME', input_shape=(128, 128, 3)),  # cnn layer
    tf.keras.layers.BatchNormalization(),  # batch norm layer
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='SAME', input_shape=(128, 128, 3)),  # cnn layer
    tf.keras.layers.BatchNormalization(),  # batch norm layer

    tf.keras.layers.MaxPooling2D(2, 2, padding='SAME'),  # pooling layer

     tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='SAME', input_shape=(128, 128, 3)),  # cnn layer
     tf.keras.layers.BatchNormalization(),  # batch norm layer
     tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='SAME', input_shape=(128, 128, 3)),  # cnn layer
     tf.keras.layers.BatchNormalization(),  # batch norm layer
     tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='SAME', input_shape=(128, 128, 3)),  # cnn layer
     tf.keras.layers.BatchNormalization(),  # batch norm layer

     tf.keras.layers.MaxPooling2D(2, 2, padding='SAME'),  # pooling layer

    # tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='SAME', input_shape=(32, 32, 3)),  # cnn layer
    # tf.keras.layers.BatchNormalization(),  # batch norm layer
    # tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='SAME', input_shape=(32, 32, 3)),  # cnn layer
    # tf.keras.layers.BatchNormalization(),  # batch norm layer
    # tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='SAME', input_shape=(32, 32, 3)),  # cnn layer
    # tf.keras.layers.BatchNormalization(),  # batch norm layer
    # tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='SAME', input_shape=(32, 32, 3)),  # cnn layer
    # tf.keras.layers.BatchNormalization(),  # batch norm layer
    #
    # tf.keras.layers.MaxPooling2D(2, 2, padding='SAME'),  # pooling layer

    # tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='SAME', input_shape=(32, 32, 3)),  # cnn layer
    # tf.keras.layers.BatchNormalization(),  # batch norm layer
    # tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='SAME', input_shape=(32, 32, 3)),  # cnn layer
    # tf.keras.layers.BatchNormalization(),  # batch norm layer
    # tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='SAME', input_shape=(32, 32, 3)),  # cnn layer
    # tf.keras.layers.BatchNormalization(),  # batch norm layer
    # tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='SAME', input_shape=(32, 32, 3)),  # cnn layer
    # tf.keras.layers.BatchNormalization(),  # batch norm layer
    #
    # tf.keras.layers.MaxPooling2D(2, 2, padding='SAME'),  # pooling layer

    tf.keras.layers.GlobalAveragePooling2D(),  # pooling layer 출력을 벡터로 변환


    # tf.keras.layers.Dense(64, activation='relu'),  # fully connected layer
    # tf.keras.layers.Dropout(0.5),
    # tf.keras.layers.Dense(128, activation='relu'),  # fully connected layer
    # tf.keras.layers.Dropout(0.5),
    # tf.keras.layers.Dense(32, activation='relu'),  # fully connected layer
    # tf.keras.layers.Dropout(0.5),
    #
    # tf.keras.layers.Dense(4, activation='softmax')  # ouput layer

    #tf.keras.layers.Dense(256, activation='relu'),  # fully connected layer
    # tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(64, activation='relu'),  # fully connected layer
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4, activation='softmax')  # output layer
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 모델 학습
EPOCH = 32
BATCH_SIZE = 16

earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',  # 모니터 기준 설정 (val loss)
                                                 patience=5,  # 5 Epoch동안 개선되지 않는다면 종료
                                                 )

reduceLR = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=3
)

data = model.fit(X_train,
                 y_train,
                 validation_data=(X_valid, y_valid),
                 epochs=EPOCH,
                 batch_size=BATCH_SIZE,
                 callbacks=[reduceLR, earlystopping]
                 )

#학습된 그래프 보기
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


# test 예측
pred_proba = model.predict(X_test)

pred_class = []

for i in pred_proba:
    pred = np.argmax(i)
    pred_class.append(pred)

pred_class = le.inverse_transform(pred_class)
print(pred_class)

