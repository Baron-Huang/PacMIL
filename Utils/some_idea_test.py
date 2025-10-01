#### Author: Dr. Pan Huang
#### Email: mrhuangpan@163.com or pan.huang@polyu.edu.hk
#### Department: PolyU, HK

import  shutil
import numpy as np
import os
import natsort
from numpy.random import shuffle
from Utils.Setup_Seed import setup_seed


if __name__ == '__main__':
    setup_seed(1)
    class_num = 'III'
    source_path = r'E:\DHM_MIL\Datasets\Cervix\Final\\'
    tar_path = r'E:\DHM_MIL\Datasets\Cervix\Final_Org\\'

    source_list = natsort.natsorted(os.listdir(source_path + class_num), alg=natsort.ns.PATH)
    source_list_len = len(source_list)
    source_list_order = np.arange(0, source_list_len)
    shuffle(source_list_order)


    train_save_order = source_list_order[:int(source_list_len * 0.6)]
    train_save_list = [source_list[i] for i in train_save_order]

    val_save_order = source_list_order[int(source_list_len * 0.6):(int(source_list_len * 0.8))]
    val_save_list = [source_list[i] for i in val_save_order]

    test_save_order = source_list_order[(int(source_list_len * 0.8)):]
    test_save_list = [source_list[i] for i in test_save_order]

    for i in train_save_list:
        shutil.copy(source_path + class_num + r'\\' + i, tar_path + 'Train' + r'\\' + class_num + r'\\' + i)

    for i in val_save_list:
        shutil.copy(source_path + class_num + r'\\' + i, tar_path + 'Val' + r'\\' + class_num + r'\\' + i)

    for i in test_save_list:
        shutil.copy(source_path + class_num + r'\\' + i, tar_path + 'Test' + r'\\' + class_num + r'\\' + i)



    print()
