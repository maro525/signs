import os
import cv2
from PIL import Image
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import hamming_loss

def img_to_matrix(filename, verbose=False):
    img = Image.open(filename).resize((70,70)).convert('L')
    img_array = np.array(img)
    return img_array

def flatten_image(img):
    s = img.shape[0] * img.shape[1]
    img_wide = []
    img_wide = img.reshape(1,s)
    return img_wide

img_dir_name = ['A','B','C','Five','Point','V']
train_images = []
train_labels = []
for f in range(6):
    train_img_dir = "Marcel-Train/" + img_dir_name[f] + "/"
    for i in os.listdir(train_img_dir):
        train_images.append(train_img_dir + i)
        train_labels.append(img_dir_name[f])

test_images = []
test_labels = []
for f in range(6):
    test_img_dir = "Marcel-Test/" + img_dir_name[f] + "/"
    for i in os.listdir(test_img_dir):
        test_images.append(test_img_dir + i)
        test_labels.append(img_dir_name[f])

train_data = []
for num in range(len(train_images)):
        img = img_to_matrix(train_images[num])
        img = flatten_image(img)
        train_data.append(img[0])
train_data = np.array(train_data)
print("train data length : " + str(len(train_data)))
print(train_data.shape)

clf = LinearSVC(C=1.0)
clf.fit(train_data, train_labels)

# test_data = []
# for num in range(len(test_images)):
#         img = img_to_matrix(test_images[num])
#         img = flatten_image(img)
#         test_data.append(img[0])
# print("test data length : " + str(len(test_data)))
# label_predict = clf.predict(test_data)
# print(hamming_loss(test_labels,label_predict))

cam = cv2.VideoCapture(0)
while True:
    ret, img = cam.read()
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img,(70,70))
    cv2.imshow('camera capture', img)
    img = flatten_image(img)
    label = clf.predict(img)
    print(label)

    cv2.imshow('cam',img)
    if cv2.waitKey(10) > 0:
        break

cam.release()
cv2.destroyAllWindows()
