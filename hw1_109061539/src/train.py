import os
import numpy as np
import datetime

from src.dataset import MNIST_dataset as Dataset
from src.model import Model

def train(path_to_train_data, path_to_val_data, path_to_checkpoints="checkpoints"):
    os.makedirs(path_to_checkpoints, exist_ok=True)
    
    print("Time: {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    train_dataset = Dataset(path_to_train_data, mode="train")
    print("load training sample: {}".format(len(train_dataset)))
    val_dataset = Dataset(path_to_val_data, mode="val")
    print("load val sample: {}".format(len(val_dataset)))

    model = Model()

    batch_size = 10
    epoch_num = 70
    should_stop = False
    best_acc = -1
    for epoch_idx in range(1, epoch_num + 1):
        
        shuffle = np.random.permutation(len(train_dataset)) # shuffle the training set
        for batch_index in range(len(train_dataset) // batch_size):
            # construct a batch of data
            data_idx = shuffle[batch_index * batch_size]
            res = train_dataset[data_idx]
            images, labels = res["image"], res["label"]
            for data_idx in [shuffle[idx] for idx in range(batch_index * batch_size + 1, (batch_index + 1) * batch_size)]:
                res = train_dataset[data_idx]
                image, label = res["image"], res["label"]

                images = np.concatenate((images, image), axis=0)
                labels = np.concatenate((labels, label), axis=0)

            # forward
            logits = model.forward(images)

            # backward and update
            model.backward(logits, labels)
            model.step()
        
        # evaluate
        val_hit = 0
        for data_idx in np.arange(len(val_dataset)):
            res = val_dataset[data_idx]
            image, label = res["image"], res["label"]

            pred = model.predict(image)
            val_hit += int(pred) == int(label[0])
        val_acc = val_hit / len(val_dataset)

        train_hit = 0
        for data_idx in np.arange(len(train_dataset)):
            res = train_dataset[data_idx]
            image, label = res["image"], res["label"]

            pred = model.predict(image)
            train_hit += int(pred) == int(label[0])
        train_acc = train_hit / len(train_dataset)

        print("Epoch {} | Train acc. {:.4f} | Val acc. {:.4f} | Lr. {:.4f}".format(epoch_idx, train_acc, val_acc, model.lr))

        if epoch_idx % 30 == 0:
            model.lr /= 2
        
        if val_acc > best_acc:
            best_acc = val_acc
            model.save(path_to_checkpoints, epoch_idx)


if __name__ == "__main__":
    train("MNIST/train", "MNIST/train")


