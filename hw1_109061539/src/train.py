import os
import numpy as np
import datetime

from src.model import Model
from src.config import TrainConfig as Config
from src.dataset import MNIST_dataset as Dataset

def train(path_to_train_data, path_to_val_data, path_to_checkpoints="checkpoints"):
    """Train a model from scratch
    Args:
        path_to_train_data: directory to train images and labels
        path_to_val_data: directory to val images and labels
        path_to_checkpoints: directory to save model weights
    Return:
        None
    """
    
    # create folder for this experiment
    time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.join(path_to_checkpoints, time), exist_ok=True)
    
    train_dataset = Dataset(path_to_train_data, mode="train")
    val_dataset = Dataset(path_to_val_data, mode="val")
    
    with open(os.path.join(path_to_checkpoints, time, "log.txt"), "a") as fp:
        fp.write("Time: {}\n".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    
        fp.write("load training sample: {}\n".format(len(train_dataset)))
        fp.write("load val sample: {}\n".format(len(val_dataset)))
    fp.close()

    model = Model()
    
    batch_size = Config.BatchSize
    epoch_num = Config.EpochNumber
    epoch_interval_to_save = Config.EpochIntervalToSave
    epoch_interval_to_decay = Config.EpochIntervalToDecay

    best_acc = -1
    for epoch_idx in range(1, epoch_num + 1):
        # shuffle the whole training set 
        shuffle = np.random.permutation(len(train_dataset))
        
        start_time = datetime.datetime.now()
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

            # backward
            model.backward(logits, labels)

            # update model weights
            model.step()
        
        spend_time = datetime.datetime.now() - start_time
        
        # compute loss and accuracy on training set and validation set
        train_hit = 0
        images = np.zeros((1, 28 * 28), dtype=np.float32)
        labels = np.zeros((1, 1), dtype=np.int32)
        for data_idx in np.arange(len(train_dataset)):
            res = train_dataset[data_idx]
            image, label = res["image"], res["label"]

            pred = model.predict(image)
            train_hit += int(pred) == int(label[0])
            
            images = np.concatenate((images, image), axis=0)
            labels = np.concatenate((labels, label), axis=0)
        logits = model.forward(images[1:])
        train_loss = model.CEloss(logits, labels[1:])
        train_acc = train_hit / len(train_dataset)

        val_hit = 0
        images = np.zeros((1, 28 * 28), dtype=np.float32)
        labels = np.zeros((1, 1), dtype=np.int32)
        for data_idx in np.arange(len(val_dataset)):
            res = val_dataset[data_idx]
            image, label = res["image"], res["label"]
            
            pred = model.predict(image)
            val_hit += int(pred) == int(label[0])

            images = np.concatenate((images, image), axis=0)
            labels = np.concatenate((labels, label), axis=0)
        logits = model.forward(images[1:])
        val_loss = model.CEloss(logits, labels[1:])
        val_acc = val_hit / len(val_dataset)
        
        # write to log file
        with open(os.path.join(path_to_checkpoints, time, "log.txt"), "a") as fp:
            fp.write("[{}] Epoch {:2d} | Train acc. {:.4f} | Train loss {:.4f} | Val acc. {:.4f} | Val loss {:.4f} | Time {}\n".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch_idx, train_acc, train_loss, val_acc, val_loss, spend_time))
        fp.close()

        if epoch_idx % epoch_interval_to_save == 0:
            # save epoch model weights
            model.save(path_to_checkpoints=os.path.join(path_to_checkpoints, time), tag=str(epoch_idx))

        if val_acc > best_acc:
            # save best model weights
            best_acc = val_acc
            model.save(path_to_checkpoints=os.path.join(path_to_checkpoints, time), tag="best")

        if epoch_idx % epoch_interval_to_decay == 0:
            # learning rate decay
            model.lr_decay()


if __name__ == "__main__":
    train(path_to_train_data="MNIST/train", path_to_val_data="MNIST/train", path_to_checkpoints="checkpoints")
