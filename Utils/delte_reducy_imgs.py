#### Author: Dr. Pan Huang
#### Email: mrhuangpan@163.com or pan.huang@polyu.edu.hk
#### Department: PolyU, HK

import  shutil
import numpy as np
import os
import natsort

if __name__ == '__main__':
    read_path = r'E:\DHM_MIL\Datasets\Cervix\Patch\III'
    save_path = r'E:\DHM_MIL\Datasets\Cervix\New_Small_Patch\III'
    read_imgs_list = natsort.natsorted(os.listdir(read_path), alg=natsort.ns.PATH)
    new_read_imgs_list = [read_imgs_list[i] for i in range(len(read_imgs_list)) if i%2 == 0]
    for i in new_read_imgs_list:
        shutil.copy(read_path + r'\\' + i, save_path + r'\\' + i)
    print()



