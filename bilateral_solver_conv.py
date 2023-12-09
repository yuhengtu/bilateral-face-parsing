import torch
from torch import nn, Tensor
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

class BilateralSolverLocal(nn.Module):
    def __init__(self,
                 sigma_space: float = 32,
                 sigma_luma: float = 8,
                 lam: float = 128,
                 kernel_size: int = 21
                 ) -> None:
        super(BilateralSolverLocal, self).__init__()
        self.sigma_space = sigma_space
        self.sigma_luma = sigma_luma
        self.lam = lam
        self.kernel_size = kernel_size

        self.w_ij = nn.Parameter(torch.FloatTensor(1, 80600960), requires_grad=False).cuda()
        self.target_binary = nn.Parameter(torch.FloatTensor(16, 448, 448), requires_grad=False).cuda()
        self.output_hair = nn.Parameter(torch.FloatTensor(16, 2, 448, 448), requires_grad=True).cuda()
        self.output_ij = nn.Parameter(torch.FloatTensor(1, 80600960), requires_grad=True).cuda()


        weight = torch.zeros((kernel_size * kernel_size - 1, 1, kernel_size, kernel_size))
        num = 0
        for i in range(kernel_size):
            for j in range(kernel_size):
                if i == j == (kernel_size - 1) // 2:
                    continue
                weight[num, 0, i, j] = -1
                weight[num, 0, (kernel_size - 1) // 2, (kernel_size - 1) // 2] = 1
                num += 1
        
        # self.conv = nn.Conv2d(
        #     1, kernel_size * kernel_size - 1, kernel_size, padding=(kernel_size - 1) // 2, padding_mode='replicate'
        # )#输入通道，        输出通道           ，卷积核       ，在高度和宽度上都添加了足够的填充
        self.conv = nn.Conv2d(1, kernel_size * kernel_size - 1, kernel_size, padding=0)
        self.conv.weight = nn.Parameter(weight.cuda(), requires_grad=False)
        self.conv.bias = nn.Parameter(torch.zeros(self.conv.weight.size()[0]).cuda(), requires_grad=False)

        image_size = (448, 448)
        position_x = torch.linspace(
            0, image_size[0]-1, image_size[0]
        )[None, :].repeat((image_size[1], 1))[None, None, :, :]
        position_y = torch.linspace(
            0, image_size[1]-1, image_size[1]
        )[:, None].repeat((1, image_size[0]))[None, None, :, :]
        # position_x: torch.Size([1, 1, 448, 448])

        position_x_ij = self.conv_ij(position_x)
        position_y_ij = self.conv_ij(position_y)
        # position_x_ij: torch.Size([1, 80600960])

        self.position_ij = - (position_x_ij ** 2 + position_y_ij ** 2) / (2 * self.sigma_space ** 2)  
        # position_ij: torch.Size([1, 80600960])

    def conv_ij(self, inp: torch.Tensor) -> torch.Tensor: 
     # inp: (batch_size, in_channels, height, width)
        inp = inp.cuda()
        batch_size = inp.shape[0]
        out = self.conv(inp)
        return out.view((batch_size, -1)) #输出展平





    def forward(self, 
                output: torch.Tensor,
                reference: torch.Tensor,
                target: torch.Tensor,) -> torch.Tensor:
                # output=out, reference=im, target=lb
        
        reference_ij = 0
        # reference：torch.Size([16, 3, 448, 448])
        for c in range(reference.shape[1]):
            for batch in range(reference.shape[0]):
                reference_c_ij = self.conv_ij(torch.Tensor(reference[batch, c, :, :])[None, None, :, :])
                reference_ij -= reference_c_ij ** 2 / (2 * self.sigma_luma ** 2)
        # reference_ij: torch.Size([1, 80600960])

        self.w_ij = torch.exp(16 * self.position_ij + reference_ij)
        # w_ij: torch.Size([1, 80600960])



        # target：torch.Size([16, 448, 448])，类别标签1-18，0是非人脸，hair的像素值是17
        target_copy = target.clone()
        target_copy[target_copy != 17] = 0 # 将像素值为 17 保留为 1，其他像素置0
        target_copy[target_copy == 17] = 1
        # self.target_binary = target.view(-1).float()
        # [1, 16*448*448]

        # self.target_binary = target.view(16, 1, -1).float()
        # [16, 448*448]

        self.target_binary = target_copy.type(torch.LongTensor).cuda()
        # print(torch.max(self.target_binary))
        # print(torch.min(self.target_binary))
        # # [16, 448, 448]
        


        # # output：torch.Size([16, 19, 448, 448])     
        # output = F.log_softmax(output, dim = 1)
        # output_hair_forconv = output[:, 17, :, :].float().view(16, 448, 448)     
        # self.output_hair = output[:, 17, :, :].float().view(16, -1)      # 深复制？ 下标17是hair？
        # # [16, 448*448]
        
        # output：torch.Size([16, 19, 448, 448])     
        output = F.log_softmax(output, dim = 1)
        output_hair_forconv = output[:, 17, :, :].float().view(16, 448, 448)
        # [16, 448, 448]
        output_hair_dim2 = output[:, 17, :, :].view(16, 1, 448, 448)
        output_hair_dim1 = 1 - output_hair_dim2
        # 将output_hair_dim1和output_hair_dim2连接到self.output_hair的第二个维度上
        self.output_hair = torch.cat((output_hair_dim1, output_hair_dim2), dim=1)
        


        # output_ij = 0
        for batch in range(output_hair_forconv.shape[0]):
            self.output_ij += self.conv_ij(torch.Tensor(output_hair_forconv[batch, :, :])[None, None, :, :])
        # output_ij: torch.Size([1, 80600960])



        # 计算损失函数
        loss_bila_1 = self.lam * torch.mean(self.w_ij * self.output_ij ** 2)
        # input：（N,C）其中C表示分类的类别数，2D损失中（N,C,H,W），或者多维损失（N,C,d1,d2,...,dk）
        # target：(N), 其中的数值在【0，c-1】之间。对于k维度的损失来说形式为（N,d1,d2,...,dk）
        loss_bila_2 = F.nll_loss(self.output_hair, self.target_binary)
        # print(loss_bila_1) # 0.3914
        # print(loss_bila_2)  # 120 六次平均
        return (loss_bila_1*300. + loss_bila_2)/24.























