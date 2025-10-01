############################# fit_functions ##############################
#### Author: Dr.Pan Huang
#### Email: panhuang@cqu.edu.cn
#### Department: COE, Chongqing University
#### Attempt: fitting functions for DHM_MIL models

########################## API Section #########################
import torch
from torch import nn
import time
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import numpy as np
from torch import optim
import pandas as pd
import random
import warnings


warnings.filterwarnings('ignore')


########################## seed_function #########################
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


########################## learning functions #########################
def cnn_lr_schedule(epoch):
    if epoch < 50:
        lr = 1e-4
    elif epoch < 75:
        lr = 2e-5
    else:
        lr = 1e-6
    return lr


def vit_lr_schedule(epoch):
    if epoch < 50:
        lr = 1e-5
    elif epoch < 75:
        lr = 5e-6
    else:
        lr = 1e-6
    return lr


def vit_lr_for_breast_schedule(epoch):
    if epoch < 50:
        lr = 2e-5
    elif epoch < 75:
        lr = 1e-5
    else:
        lr = 1e-6
    return lr


def fusion_lr_schedule(epoch):
    if epoch < 50:
        lr = 5e-5
    elif epoch < 75:
        lr = 2e-6
    else:
        lr = 1e-6
    return lr


def one_hot(org_x=None, pre_dim=3):
    one_x = np.zeros((org_x.shape[0], pre_dim))
    for i in range(org_x.shape[0]):
        one_x[i, int(org_x[i])] = 1
    return one_x


########################## testing function #########################
def testing_funnction(test_model=None, train_loader=None, val_loader=None, test_loader=None, gpu_device=0,
                      out_mode=None, proba_mode=False, proba_value=0.5, class_num = 2,
                      roc_save_path = None, bags_stat = True, bag_relations_path = None):
    loss_fn = nn.CrossEntropyLoss()
    test_model.eval()
    train_acc = []
    train_loss = []
    for train_img_list, train_label in train_loader:
        train_label = train_label.cuda(gpu_device)
        with torch.no_grad():
            train_pre_y = torch.zeros((1, class_num)).cuda(gpu_device)
            for train_img in train_img_list:
                if bags_stat == True:
                    sti_pre_y, sti_tokens_weight = test_model(train_img.cuda(gpu_device))
                else:
                    sti_pre_y = test_model(train_img.cuda(gpu_device))
                train_pre_y = torch.cat((train_pre_y, sti_pre_y))
            train_pre_y = train_pre_y[1:]
            #train_pre_y = torch.softmax(train_pre_y, dim=1)
            train_loss.append(loss_fn(train_pre_y, train_label).detach().cpu().numpy())

            if proba_mode == True:
                train_pre_y = torch.softmax(train_pre_y, dim=1)
                train_pre_label_proba = torch.argmax(train_pre_y, dim=1)
                for proba_in in range(train_pre_label_proba.shape[0]):
                    if train_pre_y[proba_in, train_pre_label_proba[proba_in]] < torch.tensor(proba_value).cuda(
                            gpu_device):
                        train_pre_label_proba[proba_in] = torch.tensor(3).cuda(gpu_device)
                train_pre_label = train_pre_label_proba
            elif proba_mode == False:
                train_pre_label = torch.argmax(train_pre_y, dim=1)
            else:
                print('error! Please select probability mode!!!')
                break

            train_acc.append(accuracy_score(train_label.detach().cpu().numpy(),
                                            train_pre_label.detach().cpu().numpy()))
    test_model.eval()
    val_acc = []
    val_loss = []
    val_all_label = np.zeros(1)
    val_all_pre_label = np.zeros(1)
    for val_img_list, val_label in val_loader:
        val_label = val_label.cuda(gpu_device)
        with torch.no_grad():
            val_pre_y = torch.zeros((1, class_num)).cuda(gpu_device)
            for val_img in val_img_list:
                if bags_stat == True:
                    val_sti_pre_y, val_sti_tokens_weight = test_model(val_img.cuda(gpu_device))
                else:
                    val_sti_pre_y = test_model(val_img.cuda(gpu_device))
                val_pre_y = torch.cat((val_pre_y, val_sti_pre_y))
            val_pre_y = val_pre_y[1:]
            # train_pre_y = torch.softmax(train_pre_y, dim=1)
            val_loss.append(loss_fn(val_pre_y, val_label).detach().cpu().numpy())

            if proba_mode == True:
                val_pre_y = torch.softmax(val_pre_y, dim=1)
                val_pre_label_proba = torch.argmax(val_pre_y, dim=1)

                for proba_in in range(val_pre_label_proba.shape[0]):
                    if val_pre_y[proba_in, val_pre_label_proba[proba_in]] < torch.tensor(proba_value).cuda(gpu_device):
                        val_pre_label_proba[proba_in] = torch.tensor(3).cuda(gpu_device)
                val_pre_label = val_pre_label_proba
            elif proba_mode == False:
                val_pre_label = torch.argmax(val_pre_y, dim=1)
            else:
                print('error! Please select probability mode!!!')
                break

            val_acc.append(accuracy_score(val_label.detach().cpu().numpy(),
                                          val_pre_label.detach().cpu().numpy()))
            val_all_label = np.concatenate((val_all_label, val_label.detach().cpu().numpy()))

            val_all_pre_label = np.concatenate((val_all_pre_label, val_pre_label.detach().cpu().numpy()))
    val_all_pre_label = val_all_pre_label[1:]
    val_all_label = val_all_label[1:]

    test_model.eval()
    test_acc = []
    test_loss = []
    test_all_label = np.zeros(1)
    test_all_pre_label = np.zeros(1)
    new_pre = np.zeros((1, class_num))
    relation_bags = {}
    count_bags = 0
    for test_img_list, test_label in test_loader:
        test_label = test_label.cuda(gpu_device)
        with torch.no_grad():
            test_pre_y = torch.zeros((1, class_num)).cuda(gpu_device)
            for test_img in test_img_list:
                count_bags += 1
                if bags_stat == True:
                    test_sti_pre_y, test_sti_tokens_weight = test_model(test_img.cuda(gpu_device))
                    #relation_bags[str(count_bags)] = np.reshape(test_sti_tokens_weight.detach().cpu().numpy(), (85))
                else:
                    test_sti_pre_y = test_model(test_img.cuda(gpu_device))
                test_pre_y = torch.cat((test_pre_y, test_sti_pre_y))
            test_pre_y = test_pre_y[1:]
            # train_pre_y = torch.softmax(train_pre_y, dim=1)
            test_loss.append(loss_fn(test_pre_y, test_label).detach().cpu().numpy())

            if proba_mode == True:
                test_pre_y = torch.softmax(test_pre_y, dim=1)
                test_pre_label_proba = torch.argmax(test_pre_y, dim=1)

                for proba_in in range(test_pre_label_proba.shape[0]):
                    if test_pre_y[proba_in, test_pre_label_proba[proba_in]] < torch.tensor(proba_value).cuda(
                            gpu_device):
                        test_pre_label_proba[proba_in] = torch.tensor(3).cuda(gpu_device)
                test_pre_label = test_pre_label_proba
            elif proba_mode == False:
                test_pre_label = torch.argmax(test_pre_y, dim=1)
            else:
                print('error! Please select probability mode!!!')
                break

            test_acc.append(accuracy_score(test_label.detach().cpu().numpy(),
                                           test_pre_label.detach().cpu().numpy()))
            test_all_label = np.concatenate((test_all_label, test_label.detach().cpu().numpy()))
            test_all_pre_label = np.concatenate((test_all_pre_label, test_pre_label.detach().cpu().numpy()))
            new_pre = np.concatenate((new_pre, test_pre_y.detach().cpu().numpy()))

    new_pre = new_pre[1:]
    test_all_label = test_all_label[1:]
    test_all_pre_label = test_all_pre_label[1:]
    test_all_proba = one_hot(org_x = test_all_label, pre_dim = class_num)
    #test_all_pre_proba = one_hot(org_x= test_all_pre_label, pre_dim = class_num)

    print('########################## testing results #########################')
    print('train_acc:{:.4}'.format(np.mean(train_acc)),
          ' val_acc:{:.4}'.format(np.mean(val_acc)),
          ' test_acc:{:.4}'.format(np.mean(test_acc)), '\n')

    print('########################## testing set results #########################')
    print(classification_report(test_all_label, test_all_pre_label, digits=4))

    # print('########################## auc results #########################')
    print(roc_auc_score(test_all_proba, new_pre))

    fpr, tpr, _ = roc_curve(np.reshape(test_all_proba, (test_all_proba.shape[0]*test_all_proba.shape[1])),
                    np.reshape(new_pre, (new_pre.shape[0]*new_pre.shape[1])))

    print(fpr.shape, tpr.shape)

    write_dict = {'fpr':fpr, 'tpr':tpr}
    roc_pd = pd.DataFrame(write_dict)
    roc_pd.to_csv(roc_save_path)

    relation_bags_pd = pd.DataFrame(relation_bags)
    relation_bags_pd.to_csv(bag_relations_path)


def view_results(mil_net=None, train_loader=None, data_parallel=False, loss_fn=None, proba_mode=False,
                 gpu_device=None, proba_value=0.85, class_num = 2, bags_stat = True):
    mil_net.eval()
    train_acc = []
    train_loss = []
    for train_img_list, train_label in train_loader:
        if data_parallel == False:
            train_label = train_label.cuda(gpu_device)
            with torch.no_grad():
                train_pre_y = torch.zeros((1, class_num)).cuda(gpu_device)
                for train_img in train_img_list:
                    if bags_stat == True:
                        sti_pre_y, _ = mil_net(train_img.cuda(gpu_device))
                    else:
                        sti_pre_y = mil_net(train_img.cuda(gpu_device))
                    train_pre_y = torch.cat((train_pre_y, sti_pre_y))
                train_pre_y = train_pre_y[1:]
                train_loss.append(loss_fn(train_pre_y, train_label).detach().cpu().numpy())
                if proba_mode == True:
                    train_pre_y = torch.softmax(train_pre_y, dim=1)
                    train_pre_label_proba = torch.argmax(train_pre_y, dim=1)
                    for proba_in in range(train_pre_label_proba.shape[0]):
                        if train_pre_y[proba_in, train_pre_label_proba[proba_in]] < \
                                torch.tensor(proba_value).cuda(gpu_device):
                            train_pre_label_proba[proba_in] = torch.tensor(3).cuda(gpu_device)
                    train_pre_label = train_pre_label_proba
                elif proba_mode == False:
                    train_pre_label = torch.argmax(train_pre_y, dim=1)
                else:
                    print('error! Please select probability mode!!!')
                    break
        else:
            train_label = train_label.cuda()
            with torch.no_grad():
                train_pre_y = torch.zeros((1, class_num)).cuda()
                for train_img in train_img_list:
                    if bags_stat == True:
                        sti_pre_y, _ = mil_net(train_img.cuda(gpu_device))
                    else:
                        sti_pre_y = mil_net(train_img.cuda(gpu_device))
                    train_pre_y = torch.cat((train_pre_y,sti_pre_y))
                train_pre_y = train_pre_y[1:]
                train_pre_y = (train_pre_y[0:int(train_pre_y.shape[0] / 2), :]
                               + train_pre_y[int(train_pre_y.shape[0] / 2):, :]) / 2
                train_loss.append(loss_fn(train_pre_y, train_label).detach().cpu().numpy())
                if proba_mode == True:
                    train_pre_y = torch.softmax(train_pre_y, dim=1)
                    train_pre_label_proba = torch.argmax(train_pre_y, dim=1)
                    for proba_in in range(train_pre_label_proba.shape[0]):
                        if train_pre_y[proba_in, train_pre_label_proba[proba_in]] < torch.tensor(proba_values).cuda():
                            train_pre_label_proba[proba_in] = torch.tensor(3).cuda()
                    train_pre_label = train_pre_label_proba
                elif proba_mode == False:
                    train_pre_label = torch.argmax(train_pre_y, dim=1)
                else:
                    print('error! Please select probability mode!!!')
                    break

        train_acc.append(accuracy_score(train_label.detach().cpu().numpy(),
                                        train_pre_label.detach().cpu().numpy()))
    return train_acc, train_loss


def view_results_parallel(mil_feature=None, mil_head=None, train_loader=None, data_parallel=False,
                          loss_fn=None, proba_mode=False, gpu_device=None, proba_value=0.85,
                          batch_size=4, bags_len=100):
    mil_feature.eval()
    mil_head.eval()
    train_acc = []
    train_loss = []
    for train_img_list, train_label in train_loader:
        train_label = train_label.cuda()
        with torch.no_grad():
            train_pre_y = torch.zeros((1, 768)).cuda()
            for train_img in train_img_list:
                train_pre_y = torch.cat((train_pre_y, mil_feature(train_img.cuda())))
            train_pre_y = train_pre_y[1:]
            train_pre_y = torch.reshape(train_pre_y,
                                        (int(train_pre_y.shape[0] / bags_len), bags_len, train_pre_y.shape[1]))
            train_pre_y = mil_head(train_pre_y)
            train_loss.append(loss_fn(train_pre_y, train_label).detach().cpu().numpy())
            if proba_mode == True:
                train_pre_y = torch.softmax(train_pre_y, dim=1)
                train_pre_label_proba = torch.argmax(train_pre_y, dim=1)
                for proba_in in range(train_pre_label_proba.shape[0]):
                    if train_pre_y[proba_in, train_pre_label_proba[proba_in]] < torch.tensor(proba_value).cuda():
                        train_pre_label_proba[proba_in] = torch.tensor(3).cuda()
                train_pre_label = train_pre_label_proba
            elif proba_mode == False:
                train_pre_label = torch.argmax(train_pre_y, dim=1)
            else:
                print('error! Please select probability mode!!!')
                break

        train_acc.append(accuracy_score(train_label.detach().cpu().numpy(),
                                        train_pre_label.detach().cpu().numpy()))
    return train_acc, train_loss, train_label, train_pre_label


def test_function_parallel(mil_feature=None, mil_head=None, train_loader=None, data_parallel=False,
                           loss_fn=None, proba_mode=False, gpu_device=None, proba_value=0.85,
                           batch_size=4, bags_len=100, val_loader=None, test_loader=None):
    loss_fn = nn.CrossEntropyLoss()
    train_acc, train_loss, _, _ = view_results_parallel(mil_feature=mil_feature, mil_head=mil_head,
                                                  train_loader=train_loader, data_parallel=data_parallel,
                                                  loss_fn=loss_fn, proba_mode=proba_mode, gpu_device=None,
                                                  proba_value=proba_value, batch_size=batch_size,
                                                  bags_len=bags_len)
    val_acc, val_loss, _, _ = view_results_parallel(mil_feature=mil_feature, mil_head=mil_head,
                                              train_loader=val_loader, data_parallel=data_parallel,
                                              loss_fn=loss_fn, proba_mode=proba_mode, gpu_device=None,
                                              proba_value=proba_value, batch_size=batch_size,
                                              bags_len=bags_len)
    test_acc, test_loss, _, _ = view_results_parallel(mil_feature=mil_feature, mil_head=mil_head,
                                                train_loader=test_loader, data_parallel=data_parallel,
                                                loss_fn=loss_fn, proba_mode=proba_mode, gpu_device=None,
                                                proba_value=proba_value, batch_size=batch_size,
                                                bags_len=bags_len)
    print('train_acc:{:.4}'.format(np.mean(train_acc)),
          ' val_acc:{:.4}'.format(np.mean(val_acc)),
          ' test_acc:{:.4}'.format(np.mean(test_acc)))
    print('train_loss:{:.4}'.format(np.mean(train_loss)),
          ' val_loss:{:.4}'.format(np.mean(val_loss)),
          ' test_loss:{:.4}'.format(np.mean(test_loss)))


########################## single-out fitting function #########################
#### ddai_net:
#### train_loader:
#### val_loader:
#### test_loader:
#### epoch:
#### gpu_device:
#### train_mode:
def Single_out_fit(mil_net=None, train_loader=None, val_loader=None, test_loader=None, proba_mode=False,
                   lr_fn=None, epoch=100, gpu_device=0, onecycle_mr=1e-2, current_lr=None, data_parallel=False,
                   weight_path=r'E:\SOTA_Model_Interpretable_Learning\SIL_Weights\Larynx\SwinT_1_.pth',
                   proba_value=0.85, class_num = 2, bags_stat = True):
    loss_fn = nn.CrossEntropyLoss()
    # ce_loss = nn.TripletMarginLoss()
    # rmp_optim = torch.optim.RMSprop(ddai_net.parameters(), lr=1e-6, weight_decay=0.0001)
    # scheduler = lr_scheduler.OneCycleLR(rmp_optim, max_lr=2e-5, epochs=500, steps_per_epoch=len(train_loader))
    # scheduler = lr_scheduler.StepLR(optimizer=rmp_optim, step_size=50, gamma=0.1)
    # torch.cuda.empty_cache()
    # torch.cuda.empty_cache()
    print('########################## training results #########################')
    if lr_fn == 'onecycle':
        rmp_optim = torch.optim.AdamW(mil_net.parameters(), lr=1e-5)
        scheduler = optim.lr_scheduler.OneCycleLR(rmp_optim, max_lr=onecycle_mr,
                                                  epochs=epoch, steps_per_epoch=len(train_loader))
    for i in range(epoch):
        start_time = time.time()
        if lr_fn == 'vit':
            rmp_optim = torch.optim.RMSprop(mil_net.parameters(), lr=vit_lr_schedule(i))
        elif lr_fn == 'cnn':
            rmp_optim = torch.optim.RMSprop(mil_net.parameters(), lr=cnn_lr_schedule(i))
        elif lr_fn == 'vit_for_breast':
            rmp_optim = torch.optim.RMSprop(mil_net.parameters(), lr=vit_lr_for_breast_schedule(i))
        elif lr_fn == 'onecycle':
            pass
        elif lr_fn == 'searching_best_lr':
            rmp_optim = torch.optim.RMSprop(mil_net.parameters(), lr=current_lr)
        else:
            print('erorr!!!!')
            return 0

        mil_net.train()
        for img_data_list, img_label in train_loader:
            if data_parallel == False:
                img_label = img_label.cuda(gpu_device)
                pre_y = torch.zeros((1, class_num)).cuda(gpu_device)
                adv_loss_sum = torch.tensor(0.0, dtype=torch.float).cuda(gpu_device)
                for img_data in img_data_list:
                    if bags_stat == True:
                        mil_pre_y, adv_loss = mil_net(img_data.cuda(gpu_device))
                        adv_loss_sum += adv_loss
                    else:
                        mil_pre_y = mil_net(img_data.cuda(gpu_device))
                    pre_y = torch.cat((pre_y,  mil_pre_y))
            else:
                img_label = img_label.cuda()
                pre_y = torch.zeros((1, class_num)).cuda()
                for img_data in img_data_list:
                    if bags_stat == True:
                        mil_pre_y, _ = mil_net(img_data.cuda(gpu_device))
                    else:
                        mil_pre_y = mil_net(img_data.cuda(gpu_device))
                    pre_y = torch.cat((pre_y, mil_pre_y))
            pre_y = pre_y[1:]

            loss_value = loss_fn(pre_y, img_label) - adv_loss_sum

            loss_value.backward()
            rmp_optim.step()
            rmp_optim.zero_grad()
        # print(ddai_net.w)
        train_acc, train_loss = view_results(mil_net=mil_net, train_loader=train_loader, data_parallel=data_parallel,
                                             loss_fn=loss_fn, proba_mode=proba_mode, gpu_device=gpu_device,
                                             proba_value=proba_value, class_num=class_num, bags_stat=bags_stat)

        val_acc, val_loss = view_results(mil_net=mil_net, train_loader=val_loader, data_parallel=data_parallel,
                                         loss_fn=loss_fn, proba_mode=proba_mode, gpu_device=gpu_device,
                                         proba_value=proba_value, class_num=class_num, bags_stat=bags_stat)

        end_time = time.time()
        print('epoch ' + str(i + 1),
              ' Time:{:.3}'.format(end_time - start_time),
              ' train_loss:{:.4}'.format(np.mean(train_loss)),
              ' train_acc:{:.4}'.format(np.mean(train_acc)),
              ' val_loss:{:.4}'.format(np.mean(val_loss)),
              ' val_acc:{:.4}'.format(np.mean(val_acc)))

        # write_1.add_scalar('train_acc',np.mean(train_acc), global_step = i)
        # write_1.add_scalar('train_loss', loss_value.detach().cpu().numpy(), global_step=i)
        # write_1.add_scalar('val_loss', val_loss.detach().cpu().numpy(), global_step=i)
        # write_1.add_scalar('val_acc', np.mean(val_acc), global_step=i)

    test_acc, test_loss = view_results(mil_net=mil_net, train_loader=test_loader, data_parallel=False,
                                       loss_fn=loss_fn, proba_mode=proba_mode, gpu_device=gpu_device,
                                       proba_value=proba_value, class_num=class_num, bags_stat=bags_stat)

    print('########################## testing results #########################')
    print('train_acc:{:.4}'.format(np.mean(train_acc)),
          ' val_acc:{:.4}'.format(np.mean(val_acc)),
          ' test_acc:{:.4}'.format(np.mean(test_acc)))

    g = mil_net.state_dict()
    torch.save(g, weight_path)

    return test_acc


########################## single-out-parallel fitting function #########################
#### ddai_net:
#### train_loader:
#### val_loader:
#### test_loader:
#### epoch:
#### gpu_device:
#### train_mode:
def Single_out_fit_parallel(mil_feature=None, mil_head=None, train_loader=None, val_loader=None, test_loader=None,
                            proba_mode=False, lr_fn=None, epoch=100, gpu_device=0, onecycle_mr=1e-2, proba_value=0.85,
                            weight_path=r'E:\SOTA_Model_Interpretable_Learning\SIL_Weights\Larynx\SwinT_1.pth',
                            batch_size=4, bags_len=100, weight_head_path=None, current_lr=None):
    loss_fn = nn.CrossEntropyLoss()
    # ce_loss = nn.TripletMarginLoss()
    # rmp_optim = torch.optim.RMSprop(ddai_net.parameters(), lr=1e-6, weight_decay=0.0001)
    # scheduler = lr_scheduler.OneCycleLR(rmp_optim, max_lr=2e-5, epochs=500, steps_per_epoch=len(train_loader))
    # scheduler = lr_scheduler.StepLR(optimizer=rmp_optim, step_size=50, gamma=0.1)
    # torch.cuda.empty_cache()
    # torch.cuda.empty_cache()
    mil_paras = [{'params': mil_feature.parameters()},
                 {'params': mil_head.parameters()}]
    print('########################## training results #########################')
    if lr_fn == 'onecycle':
        rmp_optim = torch.optim.AdamW(mil_paras, lr=1e-5)
        scheduler = optim.lr_scheduler.OneCycleLR(rmp_optim, max_lr=onecycle_mr,
                                                  epochs=epoch, steps_per_epoch=len(train_loader))
    for i in range(epoch):
        start_time = time.time()
        if lr_fn == 'vit':
            rmp_optim = torch.optim.RMSprop(mil_paras, lr=vit_lr_schedule(i))
        elif lr_fn == 'cnn':
            rmp_optim = torch.optim.RMSprop(mil_paras, lr=cnn_lr_schedule(i))
        elif lr_fn == 'vit_for_breast':
            rmp_optim = torch.optim.RMSprop(mil_paras, lr=vit_lr_for_breast_schedule(i))
        elif lr_fn == 'onecycle':
            pass
        elif lr_fn == 'searching_best_lr':
            rmp_optim = torch.optim.RMSprop(mil_paras, lr=current_lr)
        else:
            print('erorr!!!!')
            return 0

        mil_feature.train()
        mil_head.train()
        for img_data_list, img_label in train_loader:
            img_label = img_label.cuda()
            pre_y = torch.zeros((1, 768)).cuda()
            for img_data in img_data_list:
                pre_y = torch.cat((pre_y, mil_feature(img_data.cuda())))
            pre_y = pre_y[1:]
            pre_y = torch.reshape(pre_y, (int(pre_y.shape[0] / bags_len), bags_len, pre_y.shape[1]))

            pre_y = mil_head(pre_y)

            loss_value = loss_fn(pre_y, img_label)

            loss_value.backward()
            rmp_optim.step()
            rmp_optim.zero_grad()
        # print(ddai_net.w)
        train_acc, train_loss,_,_ = view_results_parallel(mil_feature=mil_feature, mil_head=mil_head,
                                                      train_loader=train_loader, loss_fn=loss_fn,
                                                      proba_mode=proba_mode, gpu_device=gpu_device,
                                                      proba_value=proba_value, batch_size=batch_size,
                                                      bags_len=bags_len)

        val_acc, val_loss,_,_ = view_results_parallel(mil_feature=mil_feature, mil_head=mil_head,
                                                  train_loader=val_loader, loss_fn=loss_fn,
                                                  proba_mode=proba_mode, gpu_device=gpu_device,
                                                  proba_value=proba_value, batch_size=batch_size,
                                                  bags_len=bags_len)

        end_time = time.time()
        print('epoch ' + str(i + 1),
              ' Time:{:.3}'.format(end_time - start_time),
              ' train_loss:{:.4}'.format(np.mean(train_loss)),
              ' train_acc:{:.4}'.format(np.mean(train_acc)),
              ' val_loss:{:.4}'.format(np.mean(val_loss)),
              ' val_acc:{:.4}'.format(np.mean(val_acc)))

        # write_1.add_scalar('train_acc',np.mean(train_acc), global_step = i)
        # write_1.add_scalar('train_loss', loss_value.detach().cpu().numpy(), global_step=i)
        # write_1.add_scalar('val_loss', val_loss.detach().cpu().numpy(), global_step=i)
        # write_1.add_scalar('val_acc', np.mean(val_acc), global_step=i)

    test_acc, test_loss,_,_ = view_results_parallel(mil_feature=mil_feature, mil_head=mil_head,
                                                train_loader=test_loader, loss_fn=loss_fn,
                                                proba_mode=proba_mode, gpu_device=gpu_device,
                                                proba_value=proba_value, batch_size=batch_size,
                                                bags_len=bags_len)

    print('########################## testing results #########################')
    print('train_acc:{:.4}'.format(np.mean(train_acc)),
          ' val_acc:{:.4}'.format(np.mean(val_acc)),
          ' test_acc:{:.4}'.format(np.mean(test_acc)))

    g = mil_feature.state_dict()
    torch.save(g, weight_path)
    g_1 = mil_head.state_dict()
    torch.save(g_1, weight_head_path)

    return test_acc


########################## base view results function #########################
#### ddai_net:
#### train_loader:
#### val_loader:
#### test_loader:
#### epoch:
#### gpu_device:
#### train_mode:
def view_results_base(base_net=None, train_loader=None, data_parallel=False, loss_fn=None, proba_mode=False,
                      gpu_device=None, proba_value=0.85, class_num = 3):
    base_net.eval()
    train_acc = []
    train_loss = []
    true_label = np.zeros(1)
    pre_label = np.zeros(1)
    pre_proba_mt = np.zeros((1, class_num))
    for train_img_list, train_label in train_loader:
        train_label = train_label.cuda(gpu_device)
        with torch.no_grad():
            train_pre_y = base_net(train_img_list.cuda(gpu_device))
            train_loss.append(loss_fn(train_pre_y, train_label).detach().cpu().numpy())
            if proba_mode == True:
                train_pre_y = torch.softmax(train_pre_y, dim=1)
                train_pre_label_proba = torch.argmax(train_pre_y, dim=1)
                for proba_in in range(train_pre_label_proba.shape[0]):
                    if train_pre_y[proba_in, train_pre_label_proba[proba_in]] < \
                            torch.tensor(proba_value).cuda(gpu_device):
                        train_pre_label_proba[proba_in] = torch.tensor(3).cuda(gpu_device)
                train_pre_label = train_pre_label_proba
            elif proba_mode == False:
                train_pre_label = torch.argmax(train_pre_y, dim=1)
            else:
                print('error! Please select probability mode!!!')
                break

            train_acc.append(accuracy_score(train_label.detach().cpu().numpy(),
                                            train_pre_label.detach().cpu().numpy()))
            true_label = np.concatenate((true_label, train_label.detach().cpu().numpy()))
            pre_label = np.concatenate((pre_label, train_pre_label.detach().cpu().numpy()))
            pre_proba_mt = np.concatenate((pre_proba_mt, train_pre_y.detach().cpu().numpy()))
    return train_acc, train_loss, true_label[1:], pre_label[1:], pre_proba_mt[1:, :]

########################## base fitting function #########################
#### ddai_net:
#### train_loader:
#### val_loader:
#### test_loader:
#### epoch:
#### gpu_device:
#### train_mode:
def fit_functions_base(base_net=None, lr_fn=None, epoch=100, gpu_device=0, onecycle_mr=1e-2, current_lr=None,
                       data_parallel=False, proba_value=0.85, class_num = 2, train_loader = None, test_loader = None,
                       weight_path=r'E:\SOTA_Model_Interpretable_Learning\SIL_Weights\Larynx\SwinT_1_.pth',
                       val_loader = None, proba_mode = None):
    loss_fn = nn.CrossEntropyLoss()
    # ce_loss = nn.TripletMarginLoss()
    # rmp_optim = torch.optim.RMSprop(ddai_net.parameters(), lr=1e-6, weight_decay=0.0001)
    # scheduler = lr_scheduler.OneCycleLR(rmp_optim, max_lr=2e-5, epochs=500, steps_per_epoch=len(train_loader))
    # scheduler = lr_scheduler.StepLR(optimizer=rmp_optim, step_size=50, gamma=0.1)
    # torch.cuda.empty_cache()
    # torch.cuda.empty_cache()
    print('########################## training results #########################')
    if lr_fn == 'onecycle':
        rmp_optim = torch.optim.AdamW(base_net.parameters(), lr=1e-5)
        scheduler = optim.lr_scheduler.OneCycleLR(rmp_optim, max_lr=onecycle_mr,
                                                  epochs=epoch, steps_per_epoch=len(train_loader))
    for i in range(epoch):
        start_time = time.time()
        if lr_fn == 'vit':
            rmp_optim = torch.optim.RMSprop(base_net.parameters(), lr=vit_lr_schedule(i))
        elif lr_fn == 'cnn':
            rmp_optim = torch.optim.RMSprop(base_net.parameters(), lr=cnn_lr_schedule(i))
        elif lr_fn == 'vit_for_breast':
            rmp_optim = torch.optim.RMSprop(base_net.parameters(), lr=vit_lr_for_breast_schedule(i))
        elif lr_fn == 'onecycle':
            pass
        elif lr_fn == 'searching_best_lr':
            rmp_optim = torch.optim.RMSprop(base_net.parameters(), lr=current_lr)
        else:
            print('erorr!!!!')
            return 0

        base_net.train()
        for img_data_list, img_label in train_loader:
            img_label = img_label.cuda(gpu_device)
            pre_y = base_net(img_data_list.cuda(gpu_device))
            loss_value = loss_fn(pre_y, img_label)
            loss_value.backward()
            rmp_optim.step()
            rmp_optim.zero_grad()
        # print(ddai_net.w)
        train_acc, train_loss, _, _, _ = view_results_base(base_net=base_net, train_loader=train_loader,
                                                  loss_fn=loss_fn, proba_mode=proba_mode, gpu_device=gpu_device,
                                                  proba_value=proba_value, class_num=class_num)

        val_acc, val_loss, _, _, _ = view_results_base(base_net=base_net, train_loader=val_loader,
                                              loss_fn=loss_fn, proba_mode=proba_mode, gpu_device=gpu_device,
                                              proba_value=proba_value, class_num=class_num)

        end_time = time.time()
        print('epoch ' + str(i + 1),
              ' Time:{:.3}'.format(end_time - start_time),
              ' train_loss:{:.4}'.format(np.mean(train_loss)),
              ' train_acc:{:.4}'.format(np.mean(train_acc)),
              ' val_loss:{:.4}'.format(np.mean(val_loss)),
              ' val_acc:{:.4}'.format(np.mean(val_acc)))

        # write_1.add_scalar('train_acc',np.mean(train_acc), global_step = i)
        # write_1.add_scalar('train_loss', loss_value.detach().cpu().numpy(), global_step=i)
        # write_1.add_scalar('val_loss', val_loss.detach().cpu().numpy(), global_step=i)
        # write_1.add_scalar('val_acc', np.mean(val_acc), global_step=i)

    test_acc, test_loss, _, _, _ =\
            view_results_base(base_net=base_net, train_loader=test_loader,
                            loss_fn=loss_fn, proba_mode=proba_mode, gpu_device=gpu_device,
                            proba_value=proba_value, class_num=class_num)

    print('########################## testing results #########################')
    print('train_acc:{:.4}'.format(np.mean(train_acc)),
          ' val_acc:{:.4}'.format(np.mean(val_acc)),
          ' test_acc:{:.4}'.format(np.mean(test_acc)))


    g = base_net.state_dict()
    torch.save(g, weight_path)

    return test_acc

def test_functions_base(base_net=None, train_loader=None, test_loader=None, val_loader=None, gpu_device=0,
                        class_num=3, proba_value=0.85, proba_mode=None, roc_save_path = None):
    loss_fn = nn.CrossEntropyLoss()
    train_acc, train_loss, _, _, _ = view_results_base(base_net=base_net, train_loader=train_loader,
                                              loss_fn=loss_fn, proba_mode=proba_mode, gpu_device=gpu_device,
                                              proba_value=proba_value, class_num=class_num)

    val_acc, val_loss, _, _, _ = view_results_base(base_net=base_net, train_loader=val_loader,
                                          loss_fn=loss_fn, proba_mode=proba_mode, gpu_device=gpu_device,
                                          proba_value=proba_value, class_num=class_num)

    test_acc, test_loss, test_label, test_pre_label, pre_proba_mt = \
            view_results_base(base_net=base_net, train_loader=test_loader,
                                loss_fn=loss_fn, proba_mode=proba_mode, gpu_device=gpu_device,
                                proba_value=proba_value, class_num=class_num)
    print('########################## testing results #########################')
    print('train_acc:{:.4}'.format(np.mean(train_acc)),
          ' val_acc:{:.4}'.format(np.mean(val_acc)),
          ' test_acc:{:.4}'.format(np.mean(test_acc)))
    print('train_loss:{:.4}'.format(np.mean(train_loss)),
          ' val_loss:{:.4}'.format(np.mean(val_loss)),
          ' test_loss:{:.4}'.format(np.mean(test_loss)))
    print('classification_report:')
    print(classification_report(test_label, test_pre_label, digits=4))
    new_true = np.zeros((test_label.shape[0], class_num))
    new_pre = np.zeros((test_label.shape[0], class_num))
    for i in range(new_true.shape[0]):
        new_true[i, int(test_label[i])] = 1

    for i in range(test_pre_label.shape[0]):
        new_pre[i, int(test_pre_label[i])] = 1
    print(roc_auc_score(y_true=np.reshape(new_true, (pre_proba_mt.shape[0] * pre_proba_mt.shape[1], 1)),
                        y_score=np.reshape(new_pre, (pre_proba_mt.shape[0] * pre_proba_mt.shape[1], 1))))
    print(roc_auc_score(y_true=np.reshape(new_true, (pre_proba_mt.shape[0] * pre_proba_mt.shape[1], 1)),
                        y_score=np.reshape(pre_proba_mt, (pre_proba_mt.shape[0] * pre_proba_mt.shape[1], 1))))

    fpr, tpr, _ = roc_curve(y_true=np.reshape(new_true, (pre_proba_mt.shape[0] * pre_proba_mt.shape[1], 1)),
                        y_score=np.reshape(pre_proba_mt, (pre_proba_mt.shape[0] * pre_proba_mt.shape[1], 1)))

    print(fpr.shape, tpr.shape)

    write_dict = {'fpr': fpr, 'tpr': tpr}
    roc_pd = pd.DataFrame(write_dict)
    roc_pd.to_csv(roc_save_path)


if __name__ == '__main__':
    max_lr = 1e-3
    min_lr = 1e-7
    max_boundary = -np.log10(max_lr)
    min_boundary = -np.log10(min_lr)

    change_log = 0
    for k in range(int(min_boundary - max_boundary) * 10):
        if k % 10 == 0:
            change_log += 1
        lr = (max_lr / (10 ** (change_log - 1))) - (k - (change_log - 1) * 10) * (max_lr / (10 ** change_log))
        print(lr)





