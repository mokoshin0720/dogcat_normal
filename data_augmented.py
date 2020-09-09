from PIL import Image
import os, glob
import numpy as np
from sklearn import model_selection

classes = ["dog", "cat"]
num_classes = len(classes)
img_size = 50
num_testdata = 96

# 画像の読み込み

X_train = []
X_test = []
Y_train = []
Y_test = []

for index, class_label in enumerate(classes):
    photos_dir = "./" + class_label
    # 各ディレクトリに含まれる画像を全てfilesに格納
    files = glob.glob(photos_dir + "/*.jpg")
    for i, file in enumerate(files):
        if i >= 480: break # 480毎以上の画像は読み込まない
        img = Image.open(file)
        img = img.convert("RGB")
        img = img.resize((img_size, img_size))
        data = np.asarray(img) # 画像をnumpy配列に変換

        if i < num_testdata:
            X_test.append(data)
            Y_test.append(index)
        else:
            for angle in range(-20, 20, 5):
                # 回転させる
                img_r = img.rotate(angle)
                data = np.asarray(img_r)
                X_train.append(data)
                Y_train.append(index)

                # 反転させる
                img_trans = img_r.transpose(Image.FLIP_LEFT_RIGHT)
                data = np.asarray(img_trans)
                X_train.append(data)
                Y_train.append(index)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(Y_train)
y_test = np.array(Y_test)

xy = (X_train, X_test, y_train, y_test)
np.save("./dog_cat_augmented.npy", xy)