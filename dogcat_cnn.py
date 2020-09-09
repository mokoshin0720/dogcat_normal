from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import keras
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

classes = ["dog", "cat"]
num_classes = len(classes)
img_size = 50

# メインの関数
def main():
    X_train, X_test, y_train, y_test = np.load("./dog_cat_normal.npy", allow_pickle = True)
    
    # データの正規化
    X_train = X_train.astype("float") / 255
    X_test = X_test.astype("float") / 255

    # 1-hotベクトル化（正解を1、間違いを0にする）
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    model = model_train(X_train, y_train)
    model_evaluate(model, X_test, y_test)

def model_train(X, y):
    model = Sequential()
    
    # 第1層目（畳み込み層）
    model.add(Conv2D(32, (3,3), padding="same", input_shape=X.shape[1:]))
    model.add(Activation("relu"))
    
    # 第2層目(畳み込み層)
    model.add(Conv2D(32, (3,3)))
    model.add(Activation("relu"))

    # 第3層目(プーリング層)
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # 第4層目（畳み込み層）
    model.add(Conv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))

    # 第5層目（畳み込み層）
    model.add(Conv2D(64, (3,3)))
    model.add(Activation("relu"))

    # 第6層目（プーリング層）
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # 第7層目（全結合層）
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    # 第8層目（出力層）
    model.add(Dense(2))
    model.add(Activation("softmax"))

    #モデルの訓練
    opt = keras.optimizers.rmsprop(lr = 0.0001, decay = 1e-6)

    model.compile(loss = "categorical_crossentropy",
                        optimizer = opt, metrics = ["accuracy"])

    model.fit(X, y, batch_size=32, epochs=100)

    #モデルの保存
    model.save("./dogcat_normal_cnn.h5")

    return model

def model_evaluate(model, X, y):
    scores = model.evaluate(X, y, verbose=1)
    print("損失：", scores[0])
    print("正確さ：", scores[1])

if __name__ == "__main__":
    main()
