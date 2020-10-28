import os
import cv2
import glob
import numpy as np


class MNIST_dataset:
    def __init__(self, path_to_data, mode):
        self.path_to_images = os.path.join(path_to_data, "images")
        self.path_to_labels = os.path.join(path_to_data, "labels")
        self.mode = mode

        self.images = []
        self.labels = []
        
        if self.mode == "test":
            # for testing set, read all 10000 data
            for filename in sorted(glob.glob(os.path.join(self.path_to_images, "*.png"))):
                img = cv2.imread(filename, 0)
                self.images.append(img)
            
            for filename in sorted(glob.glob(os.path.join(self.path_to_labels, "*.txt"))):
                with open(filename, "r") as fp:
                    line = fp.readline()
                    self.labels.append(int(line[1]))
                fp.close()
        elif self.mode == "train":
            # the first 42000 data is for training
            for filename in sorted(glob.glob(os.path.join(self.path_to_images, "*.png")))[:10000]:
                img = cv2.imread(filename, 0)
                self.images.append(img)
            
            for filename in sorted(glob.glob(os.path.join(self.path_to_labels, "*.txt")))[:10000]:
                with open(filename, "r") as fp:
                    line = fp.readline()
                    self.labels.append(int(line[1]))
                fp.close()
        elif self.mode == "val":
            # the last 18000 data is for validation
            for filename in sorted(glob.glob(os.path.join(self.path_to_images, "*.png")))[10000:15000]:
            # for filename in sorted(glob.glob(os.path.join(self.path_to_images, "*.png")))[60000 * 7 // 10:]:
                img = cv2.imread(filename, 0)
                self.images.append(img)
            
            for filename in sorted(glob.glob(os.path.join(self.path_to_labels, "*.txt")))[10000:15000]:
                with open(filename, "r") as fp:
                    line = fp.readline()
                    self.labels.append(int(line[1]))
                fp.close()

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        # assert isinstance(index, int), "Please use an single integer to fetch data"

        image = self.images[index]
        image = np.array(image, dtype=float)
        image = image.reshape(1, -1)
        image /= 255.0

        label = self.labels[index]
        label = np.array(label)
        label = label.reshape(1, -1)

        # if self.mode == "train":
            # images = self.preprocessing(images)
        
        return {"image": image, "label": label}
    
    @staticmethod
    def preprocessing(imgs):
        mean, std = 0.1307, 0.3081
        for img in imgs:
            img = (img - mean) / std
        return imgs

if __name__ == "__main__":
    mnist_dataset = MNIST_dataset(path_to_data="MNIST/test", mode="train")

    for i in np.arange(0, len(mnist_dataset), 2):
        res1 = mnist_dataset[i]
        res2 = mnist_dataset[i + 1]
        image1 = res1["image"]
        image2 = res2["image"]

        image = np.concatenate((image1, image2), axis=0)
        cv2.imshow("image", image)

        key = cv2.waitKey(1000)

    print(image.shape)
    print(label.shape)

