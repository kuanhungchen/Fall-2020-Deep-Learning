import argparse
import numpy as np

from src.dataset import Dataset
from src.model import Model 
from utils import lr_schedule, CEloss, one_hot_encoding


def train(path_to_data, path_to_checkpoints, epoch_num, batch_size, learning_rate):
    # create datasets
    train_dataset = Dataset(path_to_data=path_to_data, mode='train')
    print('[Train] load training sample: {}'.format(len(train_dataset)))
    val_dataset = Dataset(path_to_data=path_to_data, mode='val')
    print('[Train] load validation sample: {}'.format(len(val_dataset)))

    # create model
    model = Model(lr=learning_rate)
    print('[Train] model initialize successfully')
    
    epoch_num = epoch_num
    batch_size = batch_size
    best_acc = 0.0
    print('[Train] start training')
    for epoch_idx in range(1, epoch_num + 1):
        shuffle = np.random.permutation(len(train_dataset))
        training_loss, validation_loss = [], []
        for batch_idx in range(len(train_dataset) // batch_size):
            # forward
            images, labels = train_dataset[shuffle[batch_idx * batch_size : (batch_idx + 1) * batch_size]]
            logits = model.forward(images)

            # backward
            one_hot_encoded_labels = one_hot_encoding(labels)
            grad = logits - one_hot_encoded_labels
            model.backward(grad)

            # update model weight
            model.update()

        # learning rate schedule
        # model.lr = lr_schedule(epoch_idx, model.lr)

        # compute training loss and accuracy
        training_hit, training_miss = 0, 0
        for data_idx in range(len(train_dataset)):
            images, labels = train_dataset[data_idx]
            logits = model.forward(images)
            training_loss.append(CEloss(logits, labels))
            training_hit += int(np.argmax(logits[0, :])) == int(labels[0])
            training_miss += int(np.argmax(logits[0, :])) != int(labels[0])

        # compute validation loss and accuracy
        validation_hit, validation_miss = 0, 0
        for data_idx in range(len(val_dataset)):
            images, labels = val_dataset[data_idx]
            logits = model.forward(images)
            validation_loss.append(CEloss(logits, labels))
            validation_hit += int(np.argmax(logits[0, :])) == int(labels[0])
            validation_miss += int(np.argmax(logits[0, :])) != int(labels[0])
        
        # write log message
        print('[Train] Epoch: {:2d} | Train acc. {:.4f} | Train loss {:.4f} | Val acc. {:.4f} | Val loss {:.4f}'.format(
                epoch_idx,
                training_hit / (training_hit + training_miss),
                sum(training_loss) / len(training_loss),
                validation_hit / (validation_hit + validation_miss),
                sum(validation_loss) / len(validation_loss)))
        
        if validation_hit / (validation_hit + validation_miss) > best_acc:
            best_acc = validation_hit / (validation_hit + validation_miss)
            model.save(path_to_checkpoints=path_to_checkpoints, tag='best_' + str(epoch_idx))

    # save mode weights
    model.save(path_to_checkpoints=path_to_checkpoints, tag='last')
    print('[Train] finish training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', '--data_dir', default='./data', help='path to data directory')
    parser.add_argument('-c', '--checkpoints_dir', default='./checkpoints', help='path to checkpoints directory')
    parser.add_argument('-e', '--epoch_num', default=30, type=int, help='number of training epoch')
    parser.add_argument('-b', '--batch_size', default=8, type=int, help='training batch size')
    parser.add_argument('-lr', '--learning_rate', default=3e-4, type=float, help='learning rate of optimizer')
    args = parser.parse_args()

    path_to_data = args.data_dir
    path_to_checkpoints = args.checkpoints_dir
    epoch_num = args.epoch_num
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    print('Arguments:')
    print('  - path to data: {}'.format(path_to_data))
    print('  - path to checkpoints: {}'.format(path_to_checkpoints))
    print('  - epoch number: {}'.format(epoch_num))
    print('  - batch size: {}'.format(batch_size))
    print('  - learning rate: {}'.format(learning_rate))
    
    train(path_to_data=path_to_data,
          path_to_checkpoints=path_to_checkpoints,
          epoch_num=epoch_num,
          batch_size=batch_size,
          learning_rate=learning_rate)

    print('Training finished! The results are saved to {}'.format(path_to_checkpoints))
