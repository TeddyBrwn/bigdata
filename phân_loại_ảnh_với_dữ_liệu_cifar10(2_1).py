
import os
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from tensorflow.keras.optimizers.legacy import SGD
from keras.layers import BatchNormalization
from keras.layers import Dropout
import tensorflow.keras.datasets as datasets
from PIL import Image

(X_train, Y_train), (X_test, Y_test) = datasets.cifar10.load_data()

X_train.shape, Y_train.shape

X_test.shape, Y_test.shape


def load_dataset():
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY


def prep_pixels(train, test):
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    return train_norm, test_norm


def change_pixel(x):
    im = Image.open(x)
    im = im.resize((256, 256))  # 32->256
    im = im.save(x+'.png')
    return x+'.png'


def define_model():
    model = Sequential()
    model.add(Conv2D(32, (7, 7), activation='relu',
              kernel_initializer='he_uniform', padding='same', input_shape=(256, 256, 7)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (7, 7), activation='relu',
              kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (7, 7), activation='relu',
              kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (7, 7), activation='relu',
              kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (7, 7), activation='relu',
              kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (7, 7), activation='relu',
              kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def run_test_harness():
    trainX, trainY, testX, testY = load_dataset()
    trainX, testX = prep_pixels(trainX, testX)
    model = define_model()
    model.fit(trainX, trainY, epochs=100, batch_size=64, verbose=0)
    model.save('final_model_new.h5')


def test_harness(x):
    trainX, trainY, testX, testY = load_dataset()
    trainX, testX = prep_pixels(trainX, testX)
    model = load_model(x)
    _, acc = model.evaluate(testX, testY, verbose=0)
    print('> %.3f' % (acc * 100.0))


def load_image(filename):
    img = load_img(filename, target_size=(32, 32))
    img = img_to_array(img)
    img = img.reshape(1, 32, 32, 3)
    img = img.astype('float32')
    img = img / 255.0
    return img


def nhan_dien(x):
    data_class = ['truck', 'car', 'bird', 'cat',
                  'deer', 'dog', 'frog', 'horse', 'ship', 'airplane']
    dem = 0
    a = 1
    b = 2
    for i in range(10):
        if x[i] == 0:
            dem += 1
        elif x[i] == 1:
            return data_class[dem]


def run_define(x):
    img = load_image(x)
    model = load_model('final_model_new.h5')
    result = (model.predict(img) > 0.4).astype('float32')
    answer = f'Đây là {nhan_dien(result[0])}'
    print(answer)


def run(x):
    temp = change_pixel(x)
    result = run_define(temp)
    if os.path.exists(temp):
        os.remove(temp)
        return result
    return result


run('8.jpg')
