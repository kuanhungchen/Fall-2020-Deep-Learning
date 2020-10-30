import os
import cv2
import glob
import numpy as np


class MNIST_dataset:
    def __init__(self, path_to_data, mode):
        # if mode == "test":
            # self.images = np.load(os.path.join(path_to_data, "images", "test_images.npy"))
            # self.labels = np.load(os.path.join(path_to_data, "labels", "test_labels.npy"))
        # elif mode == "train":
            # self.images = np.load(os.path.join(path_to_data, "images", "train_images.npy"))[:10000]
            # self.labels = np.load(os.path.join(path_to_data, "labels", "train_labels.npy"))[:10000]
        # elif mode == "val":
            # self.images = np.load(os.path.join(path_to_data, "images", "train_images.npy"))[10000:15000]
            # self.labels = np.load(os.path.join(path_to_data, "labels", "train_labels.npy"))[10000:15000]
        
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
            for filename in sorted(glob.glob(os.path.join(self.path_to_images, "*.png")))[:42000]:
                img = cv2.imread(filename, 0)
                self.images.append(img)
            
            for filename in sorted(glob.glob(os.path.join(self.path_to_labels, "*.txt")))[:42000]:
                with open(filename, "r") as fp:
                    line = fp.readline()
                    self.labels.append(int(line[1]))
                fp.close()
        elif self.mode == "val":
            for filename in sorted(glob.glob(os.path.join(self.path_to_images, "*.png")))[42000:60000]:
                img = cv2.imread(filename, 0)
                self.images.append(img)
            
            for filename in sorted(glob.glob(os.path.join(self.path_to_labels, "*.txt")))[42000:60000]:
                with open(filename, "r") as fp:
                    line = fp.readline()
                    self.labels.append(int(line[1]))
                fp.close()

    def __len__(self):
        # return self.images.shape[0]
        return len(self.images)

    def __getitem__(self, index):
        # image = self.images[index]
        # image = image.reshape(1, -1)

        # label = self.labels[index]
        # label = label.reshape(1, -1)

        # return {"image": image, "label": label}
        

        image = self.images[index]
        image = np.array(image, dtype=np.float32)
        image = image.reshape(1, -1)
        image /= 255.0

        label = self.labels[index]
        label = np.array(label, dtype=np.int32)
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
    train_dataset = MNIST_dataset("MNIST/train", "train")
    data = train_dataset[0]
    print(type(data["image"][0][0]))
    print(type(data["label"][0][0]))
    
    print(data["image"].shape)
    print(data["label"].shape)

    print(data["image"])
    print(data["label"])
