import os

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from PIL import Image

SubFolderString = "PCD_POD_"

def generate_image_from_point_cloud(pcd_path, pcd_path2, img_path, margin_ratio=0.1, base_size=5, visualize=False):
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd_arr = np.asarray(pcd.points)
    assert np.all(pcd_arr[:, 2] == 0), 'the 3rd axis must be zero'
    image_arr = pcd_arr[:, :2]
    xs, ys = image_arr[:, 0], image_arr[:, 1]

    pcd = o3d.io.read_point_cloud(pcd_path2)
    pcd_arr = np.asarray(pcd.points)
    assert np.all(pcd_arr[:, 2] == 0), 'the 3rd axis must be zero'
    image_arr = pcd_arr[:, :2]
    xs2, ys2 = image_arr[:, 0], image_arr[:, 1]

    xmin, xmax, ymin, ymax = xs.min(), xs.max(), ys.min(), ys.max()

    x_margin = (xmax - xmin) * margin_ratio
    y_margin = (ymax - ymin) * margin_ratio

    x_width = xmax - xmin + 2 * x_margin
    y_width = ymax - ymin + 2 * y_margin

    # h_w_ratio = y_width / x_width
    h_w_ratio = 1

    plt.figure(figsize=(base_size, h_w_ratio * base_size))
    # plt.xlim(xmin - x_margin, xmax + x_margin)
    # plt.ylim(ymin - y_margin, ymax + y_margin)

    # CC
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    #AK
    plt.xlim(-3, 6)
    plt.ylim(-3, 6)

    #FOISE
    plt.xlim(-3, 6)
    plt.ylim(-3, 6)

    #HALL
    plt.xlim(-7, 16)
    plt.ylim(-20, 3)

    # # LAB
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)

    # LOUNGE
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)

    # POD
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    # setting plot style there!!
    # xs = np.concatenate((xs, xs2), axis=0)
    # xs = np.concatenate((ys, ys2), axis=0)
    plt.plot(0, 0, 'o', markersize=2, alpha=0.7)
    plt.plot(xs, ys, 'o', markersize=0.8,  alpha=0.7)
    plt.plot(xs2, ys2, 'o', markersize=0.8, alpha=0.7)
    _ = plt.axis('off')
    plt.tight_layout()
    plt.savefig(img_path, dpi=550)

    if visualize:
        plt.show()

    print(f'Image @ {img_path} !')

if __name__ == '__main__':
    i = 0
    for filename in sorted(os.listdir('data/' + SubFolderString + 'People')):
        if filename.endswith(".pcd"):
            i = i + 1
            if i > 2600:
                stripped_filename = filename[:-4]
                filenamecp = filename
                filename = 'data/' + SubFolderString + 'Enviroment/' + filenamecp
                filename2 = 'data/' + SubFolderString + 'People/' + filenamecp
                img_path = 'data/' + SubFolderString + 'Both/img/' + stripped_filename + '.jpg'
                generate_image_from_point_cloud(filename, filename2, img_path, margin_ratio=0, base_size=5, visualize=False)


    # pcd_path = '../data/PCD_ISO_NEGATIVES/1627587896_505629539_23_0.pcd'
    # img_path = '../data/NEGATIVES/image/1627587896_505629539_23_0.png'
    # img_size = 64
    # generate_image_from_point_cloud(pcd_path, img_path, margin_ratio=0.1, base_size=5, visualize=False)
    # # res = scale_image(img_path, img_size=img_size)
    # # print(res.shape)
    # assert res.shape == (img_size, img_size, 3), 'Some error!'