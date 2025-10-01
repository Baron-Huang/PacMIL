############################# pacViT_modules ##############################
#### Author: Pan Huang, PhD student
#### Email: panhuang@cqu.edu.cn
#### Department: CQU-NTU(CSC Funding)
#### Attempt: creating pacViT model by loading pretrained weight for searching the best learning rate

########################## API Section #########################
from Models.SwinT_models.models.swin_transformer import SwinTransformer
from torch import nn
import torch
from torchsummaryX import summary
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np


class LambdaSheduler(nn.Module):
	def __init__(self, gamma=1.0, max_iter=1000, **kwargs):
		super(LambdaSheduler, self).__init__()
		self.gamma = gamma
		self.max_iter = max_iter
		self.curr_iter = 0

	def lamb(self):
		p = self.curr_iter / self.max_iter
		lamb = 2. / (1. + np.exp(-self.gamma * p)) - 1
		return lamb

	def step(self):
		self.curr_iter = min(self.curr_iter + 1, self.max_iter)


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class Discriminator(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=3):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        layers = [
			nn.Linear(input_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, 1),
			nn.Sigmoid()
		]
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


###Acknowledgement: The adversarial loss implementation is inspired by http://transfer.thuml.ai/
class AdversarialLoss(nn.Module):
    def __init__(self, input_dim = 60, gamma=1.0, max_iter=1000, use_lambda_scheduler=True, **kwargs):
        super(AdversarialLoss, self).__init__()
        self.input_dim = input_dim
        self.domain_classifier = Discriminator(input_dim = self.input_dim, hidden_dim=2 * self.input_dim)
        self.use_lambda_scheduler = use_lambda_scheduler
        if self.use_lambda_scheduler:
            self.lambda_scheduler = LambdaSheduler(gamma, max_iter)

    def forward(self, source, target):
        lamb = 1.0
        if self.use_lambda_scheduler:
            lamb = self.lambda_scheduler.lamb()
            self.lambda_scheduler.step()
        source_loss = self.get_adversarial_result(source, True, lamb)
        target_loss = self.get_adversarial_result(target, False, lamb)
        adv_loss = 0.5 * (source_loss + target_loss)
        return adv_loss

    def get_adversarial_result(self, x, source=True, lamb=1.0):
        x = ReverseLayerF.apply(x, lamb)
        domain_pred = self.domain_classifier(x)
        device = domain_pred.device
        if source:
            domain_label = torch.ones(len(x), 1).long()
        else:
            domain_label = torch.zeros(len(x), 1).long()
        loss_fn = nn.BCELoss()
        loss_adv = loss_fn(domain_pred, domain_label.float().to(device))
        return loss_adv


class Pac_Block(nn.Module):
    def __init__(self, tokens_shape = None, project_scale = 2, pool_size = (9, 768)):
        super(Pac_Block, self).__init__()
        self.project_1 = nn.Linear(tokens_shape, tokens_shape * project_scale)
        self.project_2 = nn.Linear(tokens_shape * project_scale, tokens_shape)
        self.softmax = nn.Softmax()
        self.averp = nn.AvgPool2d(kernel_size=pool_size, stride=pool_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.averp(x)
        y = torch.reshape(y, (y.shape[1], y.shape[0]))
        y = self.project_1(y)
        y = self.relu(y)
        y = self.project_2(y)
        y = self.softmax(y)
        return y


class Pac_Block_Sum(nn.Module):
    def __init__(self, tokens_shape = None, project_scale = 2, pool_size = (9, 768)):
        super(Pac_Block_Sum, self).__init__()
        self.project_1 = nn.Linear(tokens_shape, tokens_shape * project_scale)
        self.project_2 = nn.Linear(tokens_shape * project_scale, tokens_shape)
        self.softmax = nn.Softmax()
        self.averp = nn.AvgPool2d(kernel_size=pool_size, stride=pool_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.averp(x)
        y = torch.reshape(y, (y.shape[0], y.shape[1]))
        y = self.project_1(y)
        y = self.relu(y)
        y = self.project_2(y)
        y = self.softmax(y)
        return y

class Pac_Block_Feb(nn.Module):
    def __init__(self, tokens_shape = None, project_scale = 2, pool_size = (9, 768)):
        super(Pac_Block_Feb, self).__init__()
        self.project_1 = nn.Linear(tokens_shape, int(tokens_shape / project_scale ))
        self.project_2 = nn.Linear(int( tokens_shape / project_scale), tokens_shape)
        self.sigmoid = nn.Sigmoid()
        self.averp = nn.AvgPool2d(kernel_size=pool_size, stride=pool_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.averp(x)
        y = torch.reshape(y, (y.shape[1], y.shape[0]))
        y = self.project_1(y)
        y = self.relu(y)
        y = self.project_2(y)
        y = self.sigmoid(y)
        return y

class PacMIL_Feb(nn.Module):
    def __init__(self, base_model=None, class_num=3, final_avp_kernel_size = 9, final_avp_tride = 9,
                 pac_token_shape = 84, pac_project_scale = 2, pac_pool_size = (9, 768), pac_lamda = 0.1):
        super(PacMIL_Feb, self).__init__()
        self.layers_0 = base_model.layers[0]
        self.layers_1 = base_model.layers[1]
        self.layers_2 = base_model.layers[2]
        self.layers_3 = base_model.layers[3]
        self.patch_embed = base_model.patch_embed
        self.pos_drop = base_model.pos_drop
        self.norm = base_model.norm
        self.head = base_model.head
        self.avgp = nn.AvgPool1d(kernel_size=final_avp_kernel_size, stride=final_avp_tride)
        self.pac_block = Pac_Block_Feb(tokens_shape = pac_token_shape, project_scale = pac_project_scale ,
                                     pool_size = pac_pool_size)
        self.lamda = pac_lamda


    def forward(self, x):
        y = self.patch_embed(x)
        y = self.pos_drop(y)
        y = self.layers_0(y)
        y = self.layers_1(y)
        y = self.layers_2(y)
        y = self.layers_3(y)
        y = self.norm(y)

        tokens_weight = self.pac_block(y)
        y = self.avgp(y.permute(0, 2, 1))
        y = torch.reshape(y, (y.shape[0], y.shape[1]))
        y = tokens_weight @ y
        y = torch.mean(y, dim=0, keepdim=True)
        y = self.head(y)
        return y, tokens_weight

class PacMIL(nn.Module):
    def __init__(self, base_model=None, class_num=3, final_avp_kernel_size = 9, final_avp_tride = 9,
                 pac_token_shape = 84, pac_project_scale = 2, pac_pool_size = (9, 768), pac_lamda = 0.1):
        super(PacMIL, self).__init__()
        self.layers_0 = base_model.layers[0]
        self.layers_1 = base_model.layers[1]
        self.layers_2 = base_model.layers[2]
        self.layers_3 = base_model.layers[3]
        self.patch_embed = base_model.patch_embed
        self.pos_drop = base_model.pos_drop
        self.norm = base_model.norm
        self.head = base_model.head
        self.avgp = nn.AvgPool1d(kernel_size=final_avp_kernel_size, stride=final_avp_tride)
        self.pac_block = Pac_Block(tokens_shape = pac_token_shape, project_scale = pac_project_scale ,
                                     pool_size = pac_pool_size)
        self.lamda = pac_lamda
        self.adv_distance = AdversarialLoss(input_dim=768)

    def forward(self, x):
        y = self.patch_embed(x)
        y = self.pos_drop(y)
        y = self.layers_0(y)
        y = self.layers_1(y)
        y = self.layers_2(y)
        y = self.layers_3(y)
        y = self.norm(y)
        tokens_weight = self.pac_block(y)
        y = self.avgp(y.permute(0, 2, 1))
        y = torch.reshape(y, (y.shape[0], y.shape[1]))
        adv_t = y[1:, :]
        adv_s = torch.zeros_like(adv_t)
        for i in range(adv_s.shape[0]):
            adv_s[i, :] = y[0, :]
        adv_dis_value = self.adv_distance(adv_s, adv_t)

        w_bags = torch.mean(y, dim=1, keepdim=True)
        y = y + self.lamda * (tokens_weight @ y)
        y = torch.mean(y, dim=0, keepdim=True)
        y = self.head(y)
        return y, adv_dis_value * 0.1


class PacMIL_for_ablation(nn.Module):
    def __init__(self, base_model=None, class_num=3):
        super(PacMIL_for_ablation, self).__init__()
        self.layers_0 = base_model.layers[0]
        self.layers_1 = base_model.layers[1]
        self.layers_2 = base_model.layers[2]
        self.layers_3 = base_model.layers[3]
        self.patch_embed = base_model.patch_embed
        self.pos_drop = base_model.pos_drop
        self.norm = base_model.norm
        self.head = base_model.head
        self.avgp = nn.AvgPool1d(kernel_size=9, stride=9)

    def forward(self, x):
        y = self.patch_embed(x)
        y = self.pos_drop(y)
        y = self.layers_0(y)
        y = self.layers_1(y)
        y = self.layers_2(y)
        y = self.layers_3(y)
        y = self.norm(y)
        y = self.avgp(y.permute(0, 2, 1))
        y = torch.reshape(y, (y.shape[0], y.shape[1]))
        w_bags = torch.mean(y, dim=1, keepdim=True)
        y = torch.mean(y, dim=0, keepdim=True)
        y = self.head(y)
        return y, w_bags

class PacMIL_Parallel_Feature(nn.Module):
    def __init__(self, base_model=None):
        super(PacMIL_Parallel_Feature, self).__init__()
        self.layers_0 = base_model.layers[0]
        self.layers_1 = base_model.layers[1]
        self.layers_2 = base_model.layers[2]
        self.layers_3 = base_model.layers[3]
        self.patch_embed = base_model.patch_embed
        self.pos_drop = base_model.pos_drop
        self.norm = base_model.norm
        self.avgp = nn.AvgPool1d(kernel_size=9, stride=9)

    def forward(self, x):
        y = self.patch_embed(x)
        y = self.pos_drop(y)
        y = self.layers_0(y)
        y = self.layers_1(y)
        y = self.layers_2(y)
        y = self.layers_3(y)
        y = self.norm(y)
        return y

class PacMIL_Parallel_Head(nn.Module):
    def __init__(self, base_model = None, class_num = 3, final_avp_kernel_size = 9, final_avp_tride = 9,
                 pac_token_shape = 85, pac_project_scale = 2, pac_pool_size = (9, 768), pac_lamda = 0.1):
        super(PacMIL_Parallel_Head, self).__init__()
        self.head = base_model.head
        self.avgp = nn.AvgPool1d(kernel_size=final_avp_kernel_size, stride=final_avp_tride)
        self.pac_block = Pac_Block_Sum(tokens_shape=pac_token_shape, project_scale=pac_project_scale,
                                       pool_size=pac_pool_size)
        self.lamda = pac_lamda
        self.adv_distance = AdversarialLoss(input_dim=768)

    def forward(self, x):
        tokens_weight = self.pac_block(x)
        print(x.shape)
        y = torch.mean(x, dim=2)
        for y_i in y:
            adv_t = y_i[1:, :]
            adv_s = torch.zeros_like(adv_t)
            for i in range(adv_s.shape[0]):
                adv_s[i, :] = y_i[0, :]
            print(adv_s.shape, adv_t.shape)
            adv_dis_value = self.adv_distance(adv_s, adv_t)

        w_bags = torch.mean(y, dim=1, keepdim=True)
        y = y + self.lamda * (tokens_weight @ y)
        y = torch.mean(y, dim=0, keepdim=True)
        y = self.head(y)
        return y, adv_dis_value * 0.1


if __name__ == '__main__':
    new_model = AdversarialLoss()
    print(new_model)



