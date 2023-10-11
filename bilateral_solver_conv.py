import torch
from torch import nn, Tensor
import numpy as np
from tqdm import tqdm

# i是滑动的1，j是依赖于i周围的-1，卷积是固定的不能被优化
class BilateralSolverLocal(nn.Module):
    def __init__(self,
                 output: torch.Tensor,
                 reference: torch.Tensor,
                 target: torch.Tensor,
                 sigma_space: float = 32,
                 sigma_luma: float = 8,
                 lam: float = 128,
                 kernel_size: int = 21
                 ) -> None:
        super().__init__()
        self.kernel_size = kernel_size

        # 创建卷积核权重矩阵
        weight = torch.zeros((kernel_size * kernel_size - 1, 1, kernel_size, kernel_size))
        # weight_16 = torch.zeros((kernel_size * kernel_size - 1, 16, kernel_size, kernel_size))
        weight = weight.cuda()
        # weight_16 = weight_16.cuda()

        num = 0
        for i in range(kernel_size):
            for j in range(kernel_size):
                if i == j == (kernel_size - 1) // 2:
                    continue
                # 设置卷积核权重为-1（中心位置为1）
                weight[num, 0, i, j] = -1
                weight[num, 0, (kernel_size - 1) // 2, (kernel_size - 1) // 2] = 1
                num += 1
        
        # self.conv = nn.Conv2d(
        #     1, kernel_size * kernel_size - 1, kernel_size, padding=(kernel_size - 1) // 2, padding_mode='replicate'
        # )#输入通道，        输出通道           ，卷积核       ，在高度和宽度上都添加了足够的填充
        self.conv = nn.Conv2d(1, kernel_size * kernel_size - 1, kernel_size, padding=0).cuda() # 使用镜像填充，将填充设置为0
        # self.conv_16 = nn.Conv2d(16, kernel_size * kernel_size - 1, kernel_size, padding=0).cuda()
        # 在conv_ij中使用nn.ReflectionPad2d来实现反射填充，或者使用nn.ReplicationPad2d来实现镜像填充

        self.conv.weight = nn.Parameter(weight.cuda(), requires_grad=False)
        # self.conv.weight_16 = nn.Parameter(weight_16.cuda(), requires_grad=False)
        self.conv.bias = nn.Parameter(torch.zeros(self.conv.weight.size()[0]).cuda(), requires_grad=False)
        # self.conv_16.bias = nn.Parameter(torch.zeros(self.conv_16.weight.size()[0]).cuda(), requires_grad=False)

        self.image_size = (reference.shape[3], reference.shape[2])

        # inp.shape,第一个元素是批量大小，第二个元素是通道数（如果是图像数据的话），然后是高度和宽度
        position_x = torch.linspace(
            0, self.image_size[0]-1, self.image_size[0]
        )[None, :].repeat((self.image_size[1], 1))[None, None, :, :]
        position_y = torch.linspace(
            0, self.image_size[1]-1, self.image_size[1]
        )[:, None].repeat((1, self.image_size[0]))[None, None, :, :]
        position_x = position_x.cuda()
        position_y = position_y.cuda()
        # position_x: torch.Size([1, 1, 448, 448])

        # 使用卷积层处理位置信息，将位置信息变换成相似性矩阵
        position_x_ij = self.conv_ij(position_x)
        position_y_ij = self.conv_ij(position_y)
        # position_x_ij: torch.Size([1, 80600960])

        position_ij = - (position_x_ij ** 2 + position_y_ij ** 2) / (2 * sigma_space ** 2)  
        # position_ij: torch.Size([1, 80600960])

        reference_ij = 0
        # if len(reference.shape) == len(target.shape):
        #     reference = reference[:, :, None]
        # 如果reference像target一样是灰度图，转换成一个带有单个颜色通道的3D图像，才能计算颜色相似矩阵

        # reference：torch.Size([16, 3, 448, 448])
        # for c in range(reference.shape[-1]):
        for c in range(reference.shape[1]):
            for batch in range(reference.shape[0]):
            # 使用卷积层处理每个通道的参考图像，将其变换成相似性矩阵
            # reference_c_ij = self.conv_ij(torch.Tensor(reference[:, :, c])[None, None, :, :])
                reference_c_ij = self.conv_ij(torch.Tensor(reference[batch, c, :, :])[None, None, :, :].cuda())
                reference_ij -= reference_c_ij ** 2 / (2 * sigma_luma ** 2)
        # reference_ij: torch.Size([1, 80600960])

        # 计算双边滤波器权重
        self.w_ij = nn.Parameter(torch.exp(16 * position_ij + reference_ij), requires_grad=False)
        # self.w_ij: torch.Size([1, 80600960])

        # target：torch.Size([16, 448, 448])
        self.target = nn.Parameter(torch.Tensor(target / 255.), requires_grad=False)

        # output：torch.Size([16, 19, 448, 448])    # 16是hair
        # self.output: torch.Size([16, 448, 448])
        self.output = nn.Parameter((torch.Tensor(output[:, 16, :, :])[:, :, :] / 255.), requires_grad=True)

        output_ij = 0
        for batch in range(output.shape[0]):
            output_ij += self.conv_ij(torch.Tensor(output[batch, 16, :, :])[None, None, :, :].cuda())
        self.output_ij = nn.Parameter(torch.Tensor(output_ij), requires_grad=True)
        # self.output_ij: torch.Size([1, 80600960])

        self.lam = lam

    def conv_ij(self, inp: torch.Tensor) -> torch.Tensor: #输入tensor输出tensor
     # (batch_size, in_channels, height, width)
        inp = inp.cuda()
        batch_size = inp.shape[0]
        out = self.conv(inp)
        # inp_with_mirror_padding = nn.ReplicationPad2d(self.kernel_size // 2)(inp)
        # out = self.conv(inp_with_mirror_padding)
        return out.view((batch_size, -1))
    #卷积层的输出展平

    # def conv_ij_16(self, inp: torch.Tensor) -> torch.Tensor: 
    #     batch_size = inp.shape[0]
    #     out = self.conv_16(inp)
    #     return out.view((batch_size, -1))

    def forward(self) -> torch.Tensor:
        # 计算损失函数
        loss = self.image_size[0] * self.image_size[1] * self.lam * torch.mean(
            self.w_ij * self.output_ij ** 2
        ) + torch.mean((self.output - self.target) ** 2)
        return loss
        # loss = self.image_size[0] * self.image_size[1] * self.lam * torch.mean(
        #     self.w_ij * self.conv_ij(self.output[None, None, :, :]) ** 2
        # ) + torch.mean((self.output - self.target) ** 2)
        # return loss


























def bilateral_solver_local(reference: np.ndarray,
                           target: np.ndarray,
                           sigma_space: float = 32,
                           sigma_luma: float = 8,
                           lam: float = 32,
                           kernel_size: int = 21, 
                           ) -> np.ndarray:
    solver = BilateralSolverLocal(reference, target, sigma_space, sigma_luma, lam, kernel_size)
    solver.cpu()

    optimizer = torch.optim.Adam(solver.parameters(), lr=1e-3)
    for _ in tqdm(range(2000)): 
        optimizer.zero_grad()
        loss = solver()
        loss.backward()
        optimizer.step()
        print(loss)

    output = torch.Tensor(solver.output).view((target.shape[0], target.shape[1])).detach().cpu().numpy() * 255

    return output



if __name__ == '__main__':
    import cv2  
    refer = np.array([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20]
        ], dtype=np.uint8)
    tgt = np.array([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20]
        ], dtype=np.uint8)

    out = bilateral_solver_local(refer, tgt, lam=1, sigma_space=32, sigma_luma=1)
    cv2.imwrite('result.png', out)


# if __name__ == '__main__':
#     # 2d
#     import cv2

#     refer = cv2.imread('reference.png')
#     tgt = cv2.imread('target.png', 0)

#     out = bilateral_solver_local(refer, tgt, lam=1, sigma_space=32, sigma_luma=1)
#     cv2.imwrite('result.png', out)

