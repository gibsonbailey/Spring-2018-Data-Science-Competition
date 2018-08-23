import csv
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

submission_data_portion = 500

with open('../test.csv', newline='') as csvfile:
    read = csv.reader(csvfile)

    i = 0
    submission_data = []
    test_data = []
    for row in read:
        del row[0]
        if(i > 0 and i <= submission_data_portion):
            submission_data.append(row)
        if i % submission_data_portion == 0:
            if i > 0:
                break
        i += 1
    # print("Finished reading training images.")


    submission_data = np.array(submission_data, dtype='f4')
    # print("Submission image shape:", submission_data.shape)

img_ind = input("input image row index:")

submission_data = submission_data.reshape(submission_data_portion, 24, 120)
plt.imshow(submission_data[int(img_ind)], cmap='gray')
plt.title("Equation 1")
plt.show()
# submission_data = submission_data.transpose([0,2,1])
# submission_data = submission_data.reshape(submission_data_portion, 5, 24, 24)
# submission_data = submission_data.transpose([0,1,3,2])
# submission_data = submission_data.reshape(submission_data_portion, 5, 576)


# print("Submission image shape:", submission_data[0].shape)
# for i in range(submission_data.shape[1]):
    # plt.imshow(submission_data[1][i], cmap='gray')
    # plt.title("Equation 2")
    # plt.show()

#
# out = []
# x1 = 1
# x2 = 2
# x3 = 1
# out.append([0,int(x1 == x2 - x3)])
# x1 = 1
# x2 = 2
# x3 = 3
# out.append([1,int(x1 + x2 == x3)])
# x1 = 5
# x2 = 2
# x3 = 3
# out.append([2,int(x1 - x2 == x3)])

# print(out)
# with open('submission.csv', 'w', newline='') as csvfile:
#     write = csv.writer(csvfile)
#     write.writerow(['index', 'label'])
#     for row in out:
#         write.writerow(row)







#
