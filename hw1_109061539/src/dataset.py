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
        
        self.images = []
        self.labels = []
        
        if self.mode == "train":
            for filename in sorted(glob.glob(os.path.join(self.path_to_images, "*.png")))[:10000]:
                img = cv2.imread(filename, 0)
                self.images.append(img)

            for filename in sorted(glob.glob(os.path.join(self.path_to_labels, "*.txt")))[:10000]:
                with open(filename, "r") as fp:
                    line = fp.readline()
                    self.labels.append(int(line[1]))
                fp.close()

        elif self.mode == "val":
            for filename in sorted(glob.glob(os.path.join(self.path_to_images, "*.png")))[10000:15000]:
                img = cv2.imread(filename, 0)
                self.images.append(img)
            
            for filename in sorted(glob.glob(os.path.join(self.path_to_labels, "*.txt")))[10000:15000]:
                with open(filename, "r") as fp:
                    line = fp.readline()
                    self.labels.append(int(line[1]))
                fp.close()

        elif self.mode == "test":
            for filename in sorted(glob.glob(os.path.join(self.path_to_images, "*.png"))):
                img = cv2.imread(filename, 0)
                self.images.append(img)
            
            for filename in sorted(glob.glob(os.path.join(self.path_to_labels, "*.txt"))):
                with open(filename, "r") as fp:
                    line = fp.readline()
                    self.labels.append(int(line[1]))
                fp.close()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        image = np.array(image, dtype=float)
        image = image.reshape(1, -1)
        image /= 255.0

        label = self.labels[index]
        label = np.array(label, dtype=int)
        label = label.reshape(1, -1)

        return {"image": image, "label": label}


if __name__ == "__main__":
    dataset = MNIST_dataset("MNIST/train", "train")
    data = dataset[0]
    img, label = data["image"], data["label"]
    print(len(dataset))
    print(img.shape)
    print(label.shape)
