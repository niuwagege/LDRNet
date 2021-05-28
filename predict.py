import tensorflow as tf
import cv2
import numpy as np

net = tf.keras.models.load_model("./model_file/model-icdar-1.4")
img = cv2.imread("./imgs/background01_datasheet001_229.png")
img = cv2.resize(img, (224, 224))
img_show = np.copy(img)
img = img.astype(np.float32)
img2 = img / 255.0
result = net([img2], training=False).numpy()[0]
coord = result[0:8]
coord = [int(x * 224) for x in coord]
cv2.circle(img_show, (coord[0], coord[1]), 3, (0, 0, 255),-1)
cv2.circle(img_show, (coord[2], coord[3]), 3, (0, 255, 255),-1)
cv2.circle(img_show, (coord[4], coord[5]), 3, (255, 0, 0),-1)
cv2.circle(img_show, (coord[6], coord[7]), 3, (0, 255, 0),-1)
cv2.imshow("show", img_show)
cv2.waitKey(0)
