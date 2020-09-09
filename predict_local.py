from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
import keras, sys, os
import numpy as np
from PIL import Image

os.environ['KMP_DUPLICATE_LIB_OK']='True'

classes = ["dog", "cat"]
num_classes = len(classes)
img_size = 50

def build_model():
    model = Sequential()
    
    # 第1層目（畳み込み層）
    model.add(Conv2D(32, (3,3), padding="same", input_shape=(50, 50, 3)))

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
    model.compile(
         optimizer="adam",
         loss="categorical_crossentropy",
         metrics=["accuracy"]
     )

     # モデルのロード
    model = load_model("./dogcat_cnn_aug.h5")

    return model

def main():
    img = Image.open(sys.argv[1])
    img = img.convert("RGB")
    img = img.resize((img_size, img_size))
    data = np.asarray(img) / 255

    X = []
    X.append(data)
    X = np.array(X)
    model = build_model()

    result = model.predict([X])[0]
    predicted = result.argmax()
    percentage = int(result[predicted] * 100)
    
    print("{0} ({1} %)".format(classes[predicted], percentage))

if __name__ == "__main__":
    main()