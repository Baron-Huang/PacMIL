#### Author: Dr. Pan Huang
#### Email: mrhuangpan@163.com or pan.huang@polyu.edu.hk
#### Department: PolyU, HK
import numpy as np
import pandas as pd

if __name__ == '__main__':
    x = np.arange(0, 1, 0.02)
    y = np.arange(0, 1, 0.02)
    write_dict = {'x':x, 'y':y}
    pd_data = pd.DataFrame(write_dict)

    pd_data.to_csv('E:\DHM_MIL\Results\ROC_data\Random.csv')
