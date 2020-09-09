from PIL import Image
import os, glob
import numpy as np
from sklearn import model_selection

classes = ["dog", "cat"]
num_classes = len(classes)
img_size = 50

# 画像の読み込み

X = []
Y = []

for index, class_label in enumerate(classes):
    photos_dir = "./" + class_label
    # 各ディレクトリに含まれる画像を全てfilesに格納
    files = glob.glob(photos_dir + "/*.jpg")
    for i, file in enumerate(files):
        if i >= 200: break # 200毎以上の画像は読み込まない
        img = Image.open(file)
        img = img.convert("RGB")
        img = img.resize((img_size, img_size))
        data = np.asarray(img) # 画像をnumpy配列に変換
        X.append(data)
        Y.append(index)

X = np.array(X)
Y = np.array(Y)

# テスト用データと学習用データを2:8に分割
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size = 0.8)
xy = (X_train, X_test, y_train, y_test)
np.save("./dog_cat_normal.npy", xy)