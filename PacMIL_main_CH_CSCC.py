############################# PacMIL_Demo ##############################
#### Author: Dr. Pan Huang
#### Email: mrhuangpan@163.com or pan.huang@polyu.edu.hk
#### Department: PolyU, HK
#### Attempt: Testing PacMIL model
#### PacMIL: An End-to-end Multi-instance Learning Network via Prior-instance Adversarial Contrastive for Cervix Pathology Grading

########################## API Section #########################
import natsort
import torch
from torch import nn
from torch.utils.data import DataLoader
import os
from Models.SwinT_models.models.swin_transformer import SwinTransformer
from Models.PacMIL_modules import PacMIL, PacMIL_Parallel_Head, PacMIL_Parallel_Feature
from Utils.training_testing_for_PacMIL import Single_out_fit, testing_funnction, Single_out_fit_parallel, test_function_parallel
from Utils.Setup_Seed import setup_seed
from Utils.Read_MIL_Datasets import Read_MIL_Datasets
import seaborn as sns
from torch.nn.parallel import DataParallel
import argparse
from Utils.load_pretrained_weight import load_swint_pretrained
from torchsummary import summary

sns.set(font='Times New Roman', font_scale=0.6)


########################## main_function #########################
if __name__ == '__main__':
    ########################## Hyparameters #########################
    paras = argparse.ArgumentParser(description='PacMIL Hyparameters')
    ##
    paras.add_argument('--bags_len', type=int, default=85)
    paras.add_argument('--batch_size', type=int, default=4)
    ##
    paras.add_argument('--random_seed', type=int, default=1)
    paras.add_argument('--gpu_device', type=int, default=0)
    paras.add_argument('--class_num', type=int, default=3)
    paras.add_argument('--epochs', type=int, default=100)
    paras.add_argument('--img_size', type=list, default=[96, 96])
    paras.add_argument('--num_workers', type=int, default=0)
    paras.add_argument('--data_parallel', type=bool, default=False)
    paras.add_argument('--parallel_gpu_ids', type=list, default=[0, 1])
    paras.add_argument('--proba_mode', type=bool, default=False)
    paras.add_argument('--proba_value', type=float, default=0.34)
    paras.add_argument('--run_mode', type=str, default='train')  #train, test, visual
    paras.add_argument('--bags_stat', type=bool, default=True)
    paras.add_argument('--roc_save_path', type=str, default='D:\PacMIL\Results\Tokens_ablation\\xxxx.csv')
    paras.add_argument('--train_read_path', type=str,
                default=r'/media/ps/e7ca49a0-dd37-483e-8750-c97d354e6c73/PacMIL/Datasets/Cervix/CH_CSCC/Train')
    paras.add_argument('--test_read_path', type=str,
                default=r'/media/ps/e7ca49a0-dd37-483e-8750-c97d354e6c73/PacMIL/Datasets/Cervix/CH_CSCC/Test')
    paras.add_argument('--val_read_path', type=str,
                default=r'/media/ps/e7ca49a0-dd37-483e-8750-c97d354e6c73/PacMIL/Datasets/Cervix/CH_CSCC/Val')
    paras.add_argument('--bag_relations_path', type=str,
                       default=r'D:\PacMIL\Results\Relation_of_bags\Larynx_new_xxx.csv')

    ###
    paras.add_argument('--weights_save_feature', type=str,
                       default=r'D:\PacMIL\Results_weight_text\PacMIL_P1.pth')
    paras.add_argument('--weights_save_head', type=str,
                      default=r'/data/HP_Projects/StiViT/Results_Weights_Text/WSI/WSI_IMG_961_96x96_Patients_Head.pth')

    ####
    paras.add_argument('--weights_save_path', type=str,
            default=r'/media/ps/e7ca49a0-dd37-483e-8750-c97d354e6c73/PacMIL/Results_weight_text/Cervix/PacMIL_CH_CSCC.pth')
    ###
    paras.add_argument('--test_weights_path', type=str,
             default=r'/media/ps/e7ca49a0-dd37-483e-8750-c97d354e6c73/PacMIL/Results_weight_text/Cervix/PacMIL_CH_CSCC.pth')

    ###
    paras.add_argument('--test_weights_feature', type=str,
                       default=r'D:\PacMIL\Results_weight_text\Larynx\Cervix_May_Ablation.pth')
    paras.add_argument('--test_weights_head', type=str,
                       default=r'/data/HP_Projects/PE_MIL/Results_Weights_Text/WSI/WSI_IMG_961_96x96_Patients_Head.pth')


    paras.add_argument('--pretrained_weights_path', type=str,
            default=r'/media/ps/e7ca49a0-dd37-483e-8750-c97d354e6c73/PacMIL/Weights/SwinT/swin_tiny_patch4_window7_224_22k.pth')


    args = paras.parse_args()
    setup_seed(args.random_seed)

    ########################## reading datas and processing datas #########################
    print('########################## reading datas and processing datas #########################')
    train_data = Read_MIL_Datasets(read_path=args.train_read_path ,img_size=args.img_size, bags_len=args.bags_len)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    test_data = Read_MIL_Datasets(read_path=args.test_read_path, img_size=args.img_size, bags_len=args.bags_len)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    val_data = Read_MIL_Datasets(read_path=args.val_read_path, img_size=args.img_size, bags_len=args.bags_len)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print('train_data:', '\n', train_data, '\n')

    ########################## creating models and visuling models #########################
    print('########################## creating models and visuling models #########################')
    swinT_base = SwinTransformer(img_size=args.img_size[0], patch_size=4, in_chans=3, num_classes=args.class_num,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=3, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False)

    ##### load pretrained weights
    checkpoint = torch.load(args.pretrained_weights_path, map_location='cpu')
    state_dict = checkpoint['model']

    load_swint_pretrained(state_dict=state_dict, swinT_base=swinT_base)

    swinT_base.load_state_dict(state_dict, strict=False)

    nn.init.trunc_normal_(swinT_base.head.weight, std=.02)
    print(swinT_base.layers[0].blocks[0].mlp.fc2.weight)

    ### creating a PacMIL model
    if args.data_parallel == False:
        pacmil_net = PacMIL(base_model=swinT_base, class_num=args.class_num, final_avp_kernel_size=9,
                            final_avp_tride=9, pac_token_shape=args.bags_len, pac_project_scale=2,
                            pac_pool_size=(9, 768), pac_lamda=0.1)
        # 计算可训练参数的数量
        #pacmil_net = PacMIL_for_ablation(base_model=swinT_base, class_num=args.class_num)
        with torch.no_grad():
            print('########################## PacMIL_summary #########################')
            #summary(pacmil_net, (3, 96, 96), device='cpu')
            print('\n', '########################## PacMIL #########################')
            print(pacmil_net, '\n')

        num_trainable_params = sum(p.numel() for p in pacmil_net.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {num_trainable_params}")

            #print(stivit_net(torch.zeros(10, 3, 96, 96)).shape)
        pacmil_net = pacmil_net.cuda(args.gpu_device)
    else:
        pacmil_feature =PacMIL_Parallel_Feature(base_model=swinT_base)
        pacmil_head = PacMIL_Parallel_Head(base_model=swinT_base, class_num=args.class_num)
        with torch.no_grad():
            print('########################## PacMIL_summary #########################')
            summary(pacmil_feature, (3, 96, 96), device='cpu')
            summary(pacmil_head, (85, 9, 768), device='cpu')
            print('\n', '########################## DHM_MILNet #########################')
            print(pacmil_feature, '\n')
            print(pacmil_head, '\n')
        pacmil_feature = pacmil_feature.cuda()
        pacmil_feature = DataParallel(pacmil_feature, device_ids=args.parallel_gpu_ids)
        pacmil_head = pacmil_head.cuda()
        pacmil_head = DataParallel(pacmil_head, device_ids=args.parallel_gpu_ids)


    ########################## fitting models and testing models #########################
    if args.run_mode == 'train':
        print('########################## fitting models and testing models #########################')
        if args.data_parallel == False:
            Single_out_fit(mil_net=pacmil_net, train_loader=train_loader, val_loader=val_loader,
                           proba_mode=args.proba_mode, test_loader=test_loader, lr_fn='vit', epoch=args.epochs,
                           gpu_device=args.gpu_device, weight_path=args.weights_save_path,
                           data_parallel=args.data_parallel, proba_value=args.proba_value,
                           class_num=args.class_num, bags_stat=args.bags_stat)
        else:
            Single_out_fit_parallel(mil_feature=pacmil_feature, mil_head=pacmil_head, train_loader=train_loader,
                                    val_loader=val_loader, proba_mode=args.proba_mode, test_loader=test_loader,
                                    lr_fn='vit', epoch=args.epochs, gpu_device=args.gpu_device,
                                    weight_path=args.weights_save_feature, proba_value=args.proba_value,
                                    bags_len=args.bags_len, batch_size=args.batch_size,
                                    weight_head_path=args.weights_save_head)


    ########################## testing function #########################
    if args.run_mode == 'test':
        print('########################## testing function #########################')
        if args.data_parallel == False:
            larynx_weight = torch.load(args.test_weights_path, map_location='cuda:0')
            pacmil_net.load_state_dict(larynx_weight, strict=True)
            testing_funnction(test_model = pacmil_net, train_loader=test_loader, val_loader=val_loader,
                              proba_value = args.proba_value, test_loader=test_loader, gpu_device=args.gpu_device,
                              out_mode = None, proba_mode=args.proba_mode, class_num=args.class_num,
                              roc_save_path = args.roc_save_path, bags_stat=args.bags_stat,
                              bag_relations_path = args.bag_relations_path)
        elif args.data_parallel == True:
            head_weight = torch.load(args.test_weights_head, map_location='cuda:0')
            feature_weight = torch.load(args.test_weights_feature, map_location='cuda:0')
            pacmil_feature.load_state_dict(feature_weight, strict=True)
            pacmil_head.load_state_dict(head_weight, strict=True)
            test_function_parallel(mil_feature= pacmil_feature, mil_head=pacmil_head, train_loader=train_loader,
                                   data_parallel=args.data_parallel, proba_mode=args.proba_mode,
                                   proba_value=args.proba_value, batch_size=args.batch_size,
                                   bags_len=args.bags_len, val_loader=val_loader, test_loader=test_loader)


    ########################## visualization function #########################
    if args.run_mode == 'visual':
        print('########################## visualization function #########################')
        from Visual_functions.PacMIL_Visualization import pacmil_visual
        dir_path = r'D:\PacMIL\Datasets\Larynx\Larynx_Org\Test\III'
        save_path = r'D:\PacMIL\Results\Relation_of_bags\New_III'
        img_path_list = natsort.natsorted(os.listdir(dir_path), alg=natsort.ns.PATH)

        for i in range(img_path_list.__sizeof__()):
            print(img_path_list[i])
            pacmil_visual(bags_weights_path=args.bag_relations_path,
                        img_path=dir_path + r'\\' + img_path_list[i],
                        save_path=save_path + r'\\' + img_path_list[i],
                        start_no=171+167, current_no=i, show_size=(768, 768))







