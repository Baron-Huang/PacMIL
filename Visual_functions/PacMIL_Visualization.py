

def pacmil_visual(bags_weights_path=None,
                img_path=r'E:\StbiViT\Datasets\Cervix\Cervix_Org\Test\I\253.jpg',
                save_path=r'E:\StbiViT\Results\Relation_of_bags\test.png',
                start_no = 0, current_no = 29, show_size = (768, 768)):
    from skimage.transform import resize
    from cv2 import addWeighted
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage import io

    bag_weights_path = bags_weights_path
    bag_weights_table = pd.read_csv(bag_weights_path)
    bag_weights_ary = bag_weights_table.to_numpy()
    bag_weights_ary = bag_weights_ary[:, 1:]
    print(bag_weights_ary[:, 0])
    img_1_bw = bag_weights_ary[:, start_no + current_no][21:]
    print(img_1_bw.shape)
    img_1_bw = img_1_bw - np.min(img_1_bw)
    img_1_bw = img_1_bw / (np.max(img_1_bw))
    img_1_bw = np.reshape(img_1_bw, (63, 1))
    add_1 = np.array(0).reshape((1, 1))
    img_1_bw = np.concatenate((img_1_bw, add_1), axis=0)
    print(img_1_bw)

    xxx = np.argsort(img_1_bw, axis=0)
    mask_1 = np.reshape(img_1_bw, (8, 8))
    print(xxx)
    print(mask_1)

    mask_1[mask_1 < 0.725] = 0
    # mask_1 = mask_1 * 255.0
    mask_1 = resize(mask_1, (show_size[0], show_size[1], 3))

    plt.figure(1)
    plt.imshow(mask_1)
    plt.xticks([])
    plt.yticks([])

    img_1 = io.imread(img_path)
    img_1 = resize(img_1, (show_size[0], show_size[1]))
    plt.figure(2)
    plt.imshow(img_1)
    plt.xticks([])
    plt.yticks([])

    interpret_img = addWeighted(img_1, 0.39, mask_1, 0.61, 0.1)
    plt.figure(3)
    plt.imshow(interpret_img)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(save_path)
    #plt.show()