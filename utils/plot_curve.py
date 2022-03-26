import matplotlib.pyplot as plt
import numpy as np
import csv
import os


def plot_curve(csv_path, smoothing=0.8):

    redundancy = '.DS_Store'
    os.system(f'find {csv_path} -name "{redundancy}" -delete')
    # plt.style.use('ggplot')
    fig = plt.figure(figsize=(6.8, 5.1))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    file_names = sorted(os.listdir(csv_path))
    for i, name in enumerate(file_names):
        x = list()
        y = list()
        file_path = os.path.join(csv_path, name)
        with open(file_path) as f:
            reader = csv.reader(f)
            for j, value in enumerate(reader):
                if j != 0:
                    x.append(int(value[1]))
                    y.append(float(value[2]))
        x = np.array(x)
        y = smooth(y, smoothing)
        y = np.array(y)

        ax.plot(x, y, label=f'layer{i}')

        # if i == 0:
        #     ax.plot(x, y, label='KC_wLac')
        # elif i == 1:
        #     ax.plot(x, y, label='KC_woLac')
        # elif i == 2:
        #     ax.plot(x, y, label='notKC_wLac')
        # else:
        #     ax.plot(x, y, label='notKC_woLac')

        # if i == 0:
        #     ax.plot(x, y, label='0.1/0.1/1/0/1')
        # elif i == 1:
        #     ax.plot(x, y, label='1/1/1/0/1')
        # elif i == 2:
        #     ax.plot(x, y, label='1/1/1/0/10')
        # elif i == 3:
        #     ax.plot(x, y, label='1/1/10/0/1')
        # else:
        #     ax.plot(x, y, label='1/1/10/0/10')

    ax.xaxis.set_label_coords(1, 0)
    ax.yaxis.set_label_coords(0, 1)
    ax.set_xlabel('step', fontsize=12)

    ax.set_ylim((29, 64))

    # ax.set_ylabel('shape mIoU', rotation='horizontal', fontsize=12)
    ax.set_ylabel('trans MSE', rotation='horizontal', fontsize=12)

    # ax.legend(loc='lower right') if 'mIoU' in ylabel else ax.legend(loc='upper right')
    ax.legend(loc='best')

    plt.show()


def smooth(scalars, weight):
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


if __name__ == '__main__':
    plot_curve('/Users/junweizheng/Desktop/curve')
