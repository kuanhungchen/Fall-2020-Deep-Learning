import numpy as np
from matplotlib import pyplot as plt


def draw(path_to_log_file):
    with open(path_to_log_file, 'r') as fp:
        for _ in range(2):
            _ = fp.readline()

        x, y = [], []
        for i in range(50000):
            line = fp.readline()
            loss = line.split('Loss:')[-1].strip()
            x.append(i)
            y.append(float(loss))
    fp.close()

    print('Show figure, len(x) = {}'.format(len(x)))
    print('First loss: {}, last loss: {}'.format(y[0], y[-1]))

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Training loss')
    plt.xticks(np.arange(min(x), max(x) + 10000, 10000))
    # plt.xticks([10000, 20000, 30000, 40000, 50000], ['10000', '20000', '30000', '40000', '50000'])
    plt.yticks([max(y), min(y)], [str(max(y)), str(min(y))])
    # plt.yticks([max(y)]
    plt.title('Training loss vs Epoch')
    plt.plot(x, y)
    plt.show()



if __name__ == '__main__':
    draw('./log_update.txt')
