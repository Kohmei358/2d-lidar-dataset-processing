import os

import numpy as np

SubFolderString = "PCD_AK_Both"


def write_ndarray_to_file(array, filename):
    with open(filename, 'w') as f:
        for item in array:
            f.write("%s\n" % int(item))


if __name__ == '__main__':

    cluster_numbers = [101, 164, 517, 535, 637, 663, 778, 1023]
    new_label = 1

    for filename in sorted(os.listdir('data/' + SubFolderString)):
        if filename.endswith(".txt"):
            initial_label = np.loadtxt('data/' + SubFolderString + '/' + filename)
            mask = np.isin(initial_label, cluster_numbers)
            initial_label[mask] = new_label
            inv_mask = np.ones(len(mask),np.bool)
            inv_mask[mask] = 0
            initial_label[inv_mask] = -1
            write_ndarray_to_file(initial_label,"data/" + SubFolderString + "/FINAL_" + filename)
