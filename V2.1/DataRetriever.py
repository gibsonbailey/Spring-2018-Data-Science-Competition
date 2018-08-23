import csv
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

# DATA_PORTION: 80,000 Maximum
# TRAIN_PORTION: 80,000 Maximum
# SUBMISSION_DATA_PORTION: 20,000 Maximum


# Retrieve data from csv files
# Returns labels in format (train_labels, eval_labels)
# Returns training data in format (train_digits, eval_digits, train_ops, eval_ops)
# Returns submission data in format (submission_data)
def retrieveLabels(filename, DATA_PORTION, TRAIN_PORTION):
    print("\n\n\nFetching data...\n\n")
    with open(filename, newline='') as csvfile:
        read = csv.reader(csvfile)

        i = 0
        train_lab = []
        test_lab = []

        train_labels_digits = []
        train_labels_ops = []

        eval_labels_digits = []
        eval_labels_ops = []

        for row in read:
            if(i > 0 and i <= TRAIN_PORTION):
                train_lab.append(row)
                if(int(row[1]) == 10 or int(row[1]) == 11 or int(row[1]) == 12):
                    train_labels_ops.append(row)
                else:
                    train_labels_digits.append(row)
            if(i > 0 and i > TRAIN_PORTION):
                test_lab.append(row)
                if(int(row[1]) == 10 or int(row[1]) == 11 or int(row[1]) == 12):
                    eval_labels_ops.append(row)
                else:
                    eval_labels_digits.append(row)
            if i % DATA_PORTION == 0:
                if i > 0:
                    break
            i += 1
        print("Finished reading " + str(DATA_PORTION) + " training labels.")

        # Change type to numpy array for use with tensorflow
        train_labels = np.asarray(train_lab, dtype=np.int32)
        eval_labels = np.asarray(test_lab, dtype=np.int32)
        train_labels_digits = np.asarray(train_labels_digits, dtype=np.int32)
        eval_labels_digits = np.asarray(eval_labels_digits, dtype=np.int32)
        train_labels_ops = np.asarray(train_labels_ops, dtype=np.int32)
        eval_labels_ops = np.asarray(eval_labels_ops, dtype=np.int32)
        # print("Train label shape:", train_labels.shape)

        return train_labels, eval_labels, train_labels_digits, train_labels_ops, eval_labels_digits, eval_labels_ops

def retrieveTrainingData(filename, train_labels, test_labels, DATA_PORTION, TRAIN_PORTION):
    with open(filename, newline='') as csvfile:
        read = csv.reader(csvfile)
        i = 0
        train_digits = []
        train_ops = []
        test_digits = []
        test_ops = []
        for row in read:
            if(i > 0 and i <= TRAIN_PORTION):
                if(train_labels[i-1][1] == 10 or train_labels[i-1][1] == 11 or train_labels[i-1][1] == 12):
                    train_ops.append(row)
                else:
                    train_digits.append(row)
            if(i > 0 and i > TRAIN_PORTION):
                if(test_labels[i-1-TRAIN_PORTION][1] == 10 or test_labels[i-1-TRAIN_PORTION][1] == 11 or test_labels[i-1-TRAIN_PORTION][1] == 12):
                    test_ops.append(row)
                else:
                    test_digits.append(row)
            if i % DATA_PORTION == 0:
                if i > 0:
                    break
            i += 1
        print("Finished reading " + str(DATA_PORTION) + " training images.")


    # Change type to numpy array for use with tensorflow
    train_digits = np.array(train_digits, dtype='f4')
    eval_digits = np.array(test_digits, dtype='f4')
    train_ops = np.array(train_ops, dtype='f4')
    eval_ops = np.array(test_ops, dtype='f4')


    # print("Digit training image shape:", train_digits.shape)
    # print("Operator training image shape:", train_ops.shape)

    # Uncomment to view training image at index of your choice
    # train_data = train_data.reshape(train_data.shape[0], 24, 24)
    # plt.imshow(train_data[20], cmap='gray')
    # plt.title(train_labels[20])
    # plt.show()

    return train_digits, eval_digits, train_ops, eval_ops

def retrieveSubmissionData(filename, SUBMISSION_DATA_PORTION):
    with open(filename, newline='') as csvfile:
        read = csv.reader(csvfile)

        i = 0
        submission_data = []
        test_data = []
        for row in read:
            del row[0]
            if(i > 0 and i <= SUBMISSION_DATA_PORTION):
                submission_data.append(row)
            if i % SUBMISSION_DATA_PORTION == 0:
                if i > 0:
                    break
            i += 1
        print("Finished reading " + str(SUBMISSION_DATA_PORTION) + " submission equation images.")


        submission_data = np.array(submission_data, dtype='f4')
        # print("Submission image shape:", submission_data.shape)

    submission_data = submission_data.reshape(SUBMISSION_DATA_PORTION, 24, 120)
    submission_data = submission_data.transpose([0,2,1])
    submission_data = submission_data.reshape(SUBMISSION_DATA_PORTION, 5, 24, 24)
    submission_data = submission_data.transpose([0,1,3,2])
    submission_data = submission_data.reshape(SUBMISSION_DATA_PORTION, 5, 576)
    # print("Submission image shape:", submission_data.shape)

    return submission_data


if __name__ == "__main__":
    DATA_PORTION = 800 # 80,000 Maximum
    TRAIN_PORTION = int(4.9 * DATA_PORTION // 5) # 80,000 Maximum
    SUBMISSION_DATA_PORTION = 2000 # 20,000 Maximum

    file_labels = '../train_labels.csv'
    train_labels, test_labels, train_labels_digits, train_labels_ops, eval_labels_digits, eval_labels_ops = retrieveLabels(file_labels, DATA_PORTION, TRAIN_PORTION)

    # print("train_labels_digits.shape:", train_labels_digits.shape)
    # print("train_labels_ops.shape:", train_labels_ops.shape)
    # print("eval_labels_digits.shape:", eval_labels_digits.shape)
    # print("eval_labels_ops.shape:", eval_labels_ops.shape)

    file_data = '../train.csv'
    t_d, e_d, t_o, e_o, = retrieveTrainingData(file_data , train_labels, test_labels, DATA_PORTION, TRAIN_PORTION)

    datafile_submission = '../test.csv'
    submission_data = retrieveSubmissionData(datafile_submission, SUBMISSION_DATA_PORTION)
