import cv2
import argparse
import numpy as np

from src.model import Model

label2text = {0: 'Carambula', 1: 'Lychee', 2: 'Pear'}

def demo(path_to_image, path_to_checkpoint):
    # load test image
    test_image = cv2.imread(path_to_image, 0)
    test_image = np.array(test_image, dtype=float)
    test_image /= 255.0
    test_image = np.reshape(test_image, (1, 32, 32, 1))
    print('[Demo] test image load successfully')

    # load model
    model = Model()
    print('[Demo] model initialize successfully')
    model.load(path_to_checkpoint=path_to_checkpoint)
    print('[Demo] model weights load successfully')

    # infer the test image
    logits = model.forward(test_image)
    prediction = int(np.argmax(logits[0, :]))
    print('[Demo] prediction: {}'.format(label2text[prediction]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--path_to_image', default='./data/Data_test/Carambula/Carambula_test_0.png', help='path to test image')
    parser.add_argument('-c', '--checkpoint', required=True, help='path to model checkpoint')
    args = parser.parse_args()

    path_to_image = args.path_to_image
    path_to_checkpoint = args.checkpoint

    print('Argument:')
    print('  - path to image: {}'.format(path_to_image))
    print('  - path to checkpoint: {}'.format(path_to_checkpoint))
    demo(path_to_image=path_to_image,
          path_to_checkpoint=path_to_checkpoint)
