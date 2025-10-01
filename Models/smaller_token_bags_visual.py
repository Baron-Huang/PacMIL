import os
import natsort
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import cv2

def small_token_bags_visul(read_path = None, bags_relate = None, save_path = None, img_name = '31.jpg'):
    img = imread(read_path + r'\\' + img_name)
    x_1 = np.array(bags_relate)
    x_4 = x_1[1:5]
    x_16 = x_1[5:21]
    x_64 = x_1[20:]

    x_4_order = np.argsort(x_4)
    x_16_order = np.argsort(x_16)
    x_64_order = np.argsort(x_64)
    test_order = x_64_order[-20:-5]
    ture_show = np.zeros((9, 9, 3), dtype=int) + 255

    plt.figure(figsize=(8, 8))
    cut_size = 8
    h_len = int(img.shape[0] / cut_size)
    w_len = int(img.shape[1] / cut_size)

    for i in range(cut_size):
        for j in range(cut_size):
            inter_img = img[(i * h_len):((i + 1) * h_len), (j * w_len):((j + 1) * w_len)]
            mask_img = np.zeros_like(inter_img)
            if ((i * 8) + j) in test_order:
                for k in range(inter_img.shape[0]):
                    for s in range(inter_img.shape[1]):
                        mask_img[k, s] = [255, 0, 0]
                mask_img = cv2.applyColorMap(mask_img, cv2.COLORMAP_JET)
                inter_img = cv2.addWeighted(inter_img, 0.8, mask_img, 0.4, 0.3)
            plt.subplot(cut_size, cut_size, ((cut_size * i) + j + 1))
            plt.imshow(inter_img)
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
    print(img_name)

    plt.savefig(save_path + r'\\' + img_name, dpi=600)
    plt.close()


if __name__ == '__main__':
    token_bags_rel = pd.read_csv(r'E:\StbiViT\Results\Relation_of_bags\test.csv')
    read_path = r'E:\StbiViT\Datasets\Cervix\Cervix_Org\Test\I'
    save_path = r'E:\StbiViT\Results\Relation_of_bags\I'
    img_name_list = natsort.natsorted(os.listdir(read_path), alg=natsort.ns.PATH)

    for i, j in enumerate(img_name_list):
        small_token_bags_visul(bags_relate = token_bags_rel[str(i+1)], img_name = j,
                             read_path= read_path, save_path= save_path)


