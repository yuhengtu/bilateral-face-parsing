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
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma_luma = sigma_luma
        self.lam = lam

        weight = torch.zeros((kernel_size * kernel_size - 1, 1, kernel_size, kernel_size))
        weight = weight

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

        self.image_size = (448, 448)

        position_x = torch.linspace(
            0, self.image_size[0]-1, self.image_size[0]
        )[None, :].repeat((self.image_size[1], 1))[None, None, :, :]
        position_y = torch.linspace(
            0, self.image_size[1]-1, self.image_size[1]
        )[:, None].repeat((1, self.image_size[0]))[None, None, :, :]
        position_x = position_x
        position_y = position_y
        # position_x: torch.Size([1, 1, 448, 448])

        position_x_ij = self.conv_ij(position_x)
        position_y_ij = self.conv_ij(position_y)
        # position_x_ij: torch.Size([1, 80600960])

        self.position_ij = - (position_x_ij ** 2 + position_y_ij ** 2) / (2 * sigma_space ** 2)  
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

        w_ij = nn.Parameter(torch.exp(16 * self.position_ij + reference_ij), requires_grad=False)
        # w_ij: torch.Size([1, 80600960])


        # # target：torch.Size([16, 448, 448])，类别标签1-18，0是非人脸，hair的像素值是17
        # target_binary = torch.zeros_like(target)
        # target_binary[target == 17] = 1 # 将像素值为 17 保留为 1，其他像素置0
        # target_binary = target_binary.view(16, -1).float()
        # target_binary = nn.Parameter(torch.Tensor(target_binary), requires_grad=False)
        # print(f'target_binary{target_binary.shape}')
        # # [16, 448*448]

        # target：torch.Size([16, 448, 448])，类别标签1-18，0是非人脸，hair的像素值是17
        target_binary = torch.zeros_like(target)
        target_binary[target == 17] = 1 # 将像素值为 17 保留为 1，其他像素置0
        target_binary = nn.Parameter(torch.Tensor(target_binary.float()), requires_grad=False)
        # [16, 448, 448]



        # # output：torch.Size([16, 19, 448, 448])     
        # output = F.softmax(output, dim = 1)
        # hair_prob = output[:, 17, :, :].float().view(16, -1)      # 深复制？ 下标17是hair？
        # hair_prob = nn.Parameter(torch.Tensor(hair_prob), requires_grad=True)
        # print(f'hair_prob{hair_prob.shape}')
        # # [16, 448*448]
        
        # output：torch.Size([16, 19, 448, 448])     
        output = F.softmax(output, dim = 1)
        output = output[:, 17, :, :].float().view(16, 448, 448)      # 深复制？ 下标17是hair？
        output = nn.Parameter(torch.Tensor(output), requires_grad=True)
        # [16, 448, 448]
        

        output_ij = 0
        for batch in range(output.shape[0]):
            output_ij += self.conv_ij(torch.Tensor(output[batch, :, :])[None, None, :, :])
        output_ij = nn.Parameter(torch.Tensor(output_ij), requires_grad=True)
        # output_ij: torch.Size([1, 80600960])



        # 计算损失函数
        loss_bila_1 = self.lam * torch.mean(w_ij * output_ij ** 2)
        loss_bila_2 = F.cross_entropy(target_binary, output)
        # print(loss_bila_1) # 0.3914
        # print(loss_bila_2)  # 120 六次平均
        return (loss_bila_1*300. + loss_bila_2)/24.























