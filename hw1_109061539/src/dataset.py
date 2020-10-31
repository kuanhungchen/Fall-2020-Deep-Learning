import os
import cv2
import glob
import numpy as np


class MNIST_dataset:
    """Dataloader for MNIST dataset

    Data folder structure:

    + --- checkpoints
    | 
    + --- MNIST -+- train -+- images --- *.png
    |            |         |
    |            |         +-- labels --- *.txt
    |            |
    |            +-- test -+- images --- *.png
    |                      |
    |                      +-- labels --- *.txt
    + --- src
    
    """
    def __init__(self, path_to_data, mode):
        assert mode in ["train", "val", "test"], "mode should be train, val or test"
        self.mode = mode

        self.path_to_images = os.path.join(path_to_data, "images")
        self.path_to_labels = os.path.join(path_to_data, "labels")
        
        self.images = np.zeros((1, 28 * 28))
        self.labels = np.zeros((1, 1))
        
        if self.mode == "train":
            for filename in sorted(glob.glob(os.path.join(self.path_to_images, "*.png")))[:42000]:
                img = cv2.imread(filename, 0)
                img = np.array(img, dtype=np.float32)
                img = img.reshape(1, -1)
                img /= 255.0

                self.images = np.concatenate((self.images, img), axis=0)
            self.images = self.images[1:]

            for filename in sorted(glob.glob(os.path.join(self.path_to_labels, "*.txt")))[:42000]:
                with open(filename, "r") as fp:
                    line = fp.readline()
                fp.close()

                label = np.array(int(line[1]), dtype=np.int32)
                label = label.reshape(1, -1)
                self.labels = np.concatenate((self.labels, label), axis=0)
            self.labels = self.labels[1:]

        elif self.mode == "val":
            for filename in sorted(glob.glob(os.path.join(self.path_to_images, "*.png")))[42000:]:
                img = cv2.imread(filename, 0)
                img = np.array(img, dtype=np.float32)
                img = img.reshape(1, -1)
                img /= 255.0

                self.images = np.concatenate((self.images, img), axis=0)
            self.images = self.images[1:]
            
            for filename in sorted(glob.glob(os.path.join(self.path_to_labels, "*.txt")))[42000:]:
                with open(filename, "r") as fp:
                    line = fp.readline()
                fp.close()

                label = np.array(int(line[1]), dtype=np.int32)
                label = label.reshape(1, -1)
                self.labels = np.concatenate((self.labels, label), axis=0)
            self.labels = self.labels[1:]

        elif self.mode == "test":
            for filename in sorted(glob.glob(os.path.join(self.path_to_images, "*.png"))):
                img = cv2.imread(filename, 0)
                img = np.array(img, dtype=np.float32)
                img = img.reshape(1, -1)
                img /= 255.0

                self.images = np.concatenate((self.images, img), axis=0)
            self.images = self.images[1:]
            
            for filename in sorted(glob.glob(os.path.join(self.path_to_labels, "*.txt"))):
                with open(filename, "r") as fp:
                    line = fp.readline()
                fp.close()

                label = np.array(int(line[1]), dtype=np.int32)
                label = label.reshape(1, -1)
                self.labels = np.concatenate((self.labels, label), axis=0)
            self.labels = self.labels[1:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        image = image.reshape(1, -1)
        label = self.labels[index]
        label = label.reshape(1, -1)

        return {"image": image, "label": label}


if __name__ == "__main__":
    dataset = MNIST_dataset("MNIST/test", "test")
    data = dataset[0]
    img, label = data["image"], data["label"]
    print(len(dataset))
    print(img.shape)
    print(label.shape)
