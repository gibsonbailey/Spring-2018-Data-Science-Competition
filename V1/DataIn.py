import csv
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

data_portion = 1000

with open('train.csv', newline='') as csvfile:
    read = csv.reader(csvfile)

    i = 0
    train_data = []
    test_data = []
    for row in read:
        del row[0]
        if(i > 0 and i <= 4 * data_portion // 5):
            train_data.append(row)
        if(i > 0 and i > 4 * data_portion // 5):
            test_data.append(row)
        if i % data_portion == 0:
            if i > 0:
                break
        i += 1
    print("Finished reading training images.")

    train_imgs = []

    for im in train_data:
        train_imgs.append(list(zip(*(iter(im),)*24)))

    test_imgs = []
    for im in test_data:
        test_imgs.append(list(zip(*(iter(im),)*24)))

    train_data = np.array(train_imgs, dtype='f4')
    eval_data = np.array(test_imgs, dtype='f4')
    print("Training image shape:", train_data.shape)
    print("Testing image shape:", eval_data.shape)


with open('train_labels.csv', newline='') as csvfile:
    read = csv.reader(csvfile)

    i = 0
    train_lab = []
    test_lab = []
    for row in read:
        if(i > 0 and i <= 4 * data_portion // 5):
            train_lab.append(row[1])
        if(i > 0 and i > 4 * data_portion // 5):
            test_lab.append(row[1])
        if i % data_portion == 0:
            if i > 0:
                break
        i += 1
    print("Finished reading training labels.")

    train_labels = np.asarray(train_lab, dtype=np.int32)
    eval_labels = np.asarray(test_lab, dtype=np.int32)
    print(train_labels.shape)

    plt.imshow(train_data[35], cmap='gray')
    plt.title(train_labels[35])
    plt.show()
