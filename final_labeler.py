import os

import numpy as np

SubFolderString = "PCD_AK_Both"


def write_ndarray_to_file(array, filename):
    with open(filename, 'w') as f:
        for item in array:
            f.write("%s\n" % int(item))


if __name__ == '__main__':

    cluster_numbers = [[101, 164, 517, 535, 637, 663, 778, 1023], #Person 0
                       [86, 88, 856], #Person 1
                       [70, 91, 251, 267, 323, 340, 383, 415, 519, 746, 817, 840, 898, 916, 953, 978, 650, 219],#2
                       [42, 421, 554, 669, 697, 814, 833, 834, 888], #3
                       [5, 82], #4
                       [651, 685, 691, 707, 718, 732, 751, 770, 732, 825, 858],#5
                       [579, 658, 749, 788, 838, 895, 933, 952, 967],#6
                       [703, 736, 776, 904], #7
                       [660, 813, 766, 925, 974]] #8
    file_number = 0
    for filename in (sorted(os.listdir('data/' + SubFolderString))):
        if filename.endswith(".txt") and "FINAL" not in filename:
            initial_label = np.loadtxt('data/' + SubFolderString + '/' + filename)
            final_label = np.full(initial_label.size, -1)
            for index in range(len(cluster_numbers)):
                # print(str(index) + " "+str(file_number))
                if index == 3 and file_number == 700:
                    cluster_numbers[index].remove(833)
                if index == 3 and file_number == 734:
                    cluster_numbers[index].remove(834)
                if index == 5 and file_number == 662:
                    cluster_numbers[index].remove(707)
                if index == 5 and file_number == 690:
                    cluster_numbers[index].remove(751)
                if index == 5 and file_number == 725:
                    cluster_numbers[index].remove(858)
                if index == 6 and file_number == 700:
                    cluster_numbers[index].remove(788)
                if index == 6 and file_number == 691:
                    cluster_numbers[index].append(751)
                if index == 6 and file_number == 686:
                    cluster_numbers[index].append(697)
                if index == 7 and file_number == 700:
                    cluster_numbers[index].append(833)
                if index == 7 and file_number == 734:
                    cluster_numbers[index].append(834)
                if index == 8 and file_number == 700:
                    cluster_numbers[index].append(788)
                mask = np.isin(initial_label, cluster_numbers[index])
                final_label[mask] = index
                write_ndarray_to_file(final_label, "data/" + SubFolderString + "/FINAL_" + filename)
            file_number = file_number + 1
