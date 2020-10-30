import cv2
import glob
import numpy as np


all_images = np.zeros((1, 784), dtype=float)

for img_path in glob.glob("MNIST/test/images/*.png"):
    img = cv2.imread(img_path, 0)
    img = np.array(img, dtype=float)
    img = img.reshape(1, -1)
    img /= 255.0

    all_images = np.concatenate((all_images, img), axis=0)
    print(all_images.shape)

np.save("MNIST/test/images/test_images.npy", all_images[1:])
