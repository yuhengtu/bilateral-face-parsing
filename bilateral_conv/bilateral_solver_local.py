import torch
from torch import nn, Tensor
import numpy as np
from tqdm import tqdm

# i是滑动的1，j是依赖于i周围的-1，卷积是固定的不能被优化
class BilateralSolverLocal(nn.Module):
    def __init__(self,
                 reference: np.ndarray,
                 target: np.ndarray,
                 sigma_space: float = 32,
                 siegma_luma: float = 8,
                 lam: float = 128,
                 kernel_size: int = 21
                 ) -> None:
        super().__init__()

        # 创建卷积核权重矩阵
        weight = torch.zeros((kernel_size * kernel_size - 1, 1, kernel_size, kernel_size))
        num = 0
        for i in range(kernel_size):
            for j in range(kernel_size):
                if i == j == (kernel_size - 1) // 2:
                    continue
                # 设置卷积核权重为-1（中心位置为1）
                weight[num, 0, i, j] = -1
                weight[num, 0, (kernel_size - 1) // 2, (kernel_size - 1) // 2] = 1
                num += 1
        # weight torch.Size([8, 1, 3, 3])        
        
        self.conv = nn.Conv2d(
            1, kernel_size * kernel_size - 1, kernel_size, padding=(kernel_size - 1) // 2, padding_mode='replicate'
        )#输入通道，        输出通道           ，卷积核       ，在高度和宽度上都添加了足够的填充
        self.conv.weight = nn.Parameter(weight, rquires_grad=False)

        # 获取图像大小
        self.image_size = (reference.shape[1], reference.shape[0])

        # 创建x和y方向上的位置信息
        position_x = torch.linspace(
            0, self.image_size[0]-1, self.image_size[0]
        )[None, :].repeat((self.image_size[1], 1))[None, None, :, :]
        # torch.Size([1, 1, 4, 5])
        position_y = torch.linspace(
            0, self.image_size[1]-1, self.image_size[1]
        )[:, None].repeat((1, self.image_size[0]))[None, None, :, :]
        # torch.Size([1, 1, 4, 5])

        # inp.shape,第一个元素是批量大小，第二个元素是通道数（如果是图像数据的话），然后是高度和宽度
        # 使用卷积层处理位置信息，将位置信息变换成相似性矩阵
        position_x_ij = self.conv_ij(position_x)
        # torch.Size([1, 160])

        position_y_ij = self.conv_ij(position_y)
        # torch.Size([1, 160])

        position_ij = - (position_x_ij ** 2 + position_y_ij ** 2) / (2 * sigma_space ** 2)   
        # torch.Size([1, 160])

        reference_ij = 0
        print(reference.shape)
        #（4,5）
        if len(reference.shape) == len(target.shape):
            reference = reference[:, :, None]
            #（4,5,1）
        for c in rangae(reference.shape[-1]):
            # 使用卷积层处理每个通道的参考图像，将其变换成相似性矩阵
            reference_c_ij = self.conv_ij(torch.Tensor(reference[:, :, c])[None, None, :, :])
            reference_ij -= reference_c_ij ** 2 / (2 * sigma_luma ** 2)

        # 计算双边滤波器权重
        self.w_ij = nn.Parameter(torch.exp(position_ij + reference_ij), requires_grad=False)
        # torch.Size([1, 160])

        self.target = nn.Parameter(torch.Tensor(target / 255.), requires_grad=False)
        #torch.Size([4, 5])
        self.output = nn.Parameter(torch.Tensor(target / 255.), requires_grad=True)
        self.lam = lam

    def conv_ij(self, inp: Tensor) -> Tensor: #输入tensor输出tensor
        batch_size = inp.shape[0]
        out = self.conv(inp)
        return out.view((batch_size, -1))
    #将卷积层的输出展平

    def forward(self) -> Tensor:
        # 计算损失函数
        loss = self.image_size[0] * self.image_size[1] * self.lam * torch.mean(
            self.w_ij * self.conv_ij(self.output[None, None, :, :]) ** 2
        ) + torch.mean((self.output - self.target) ** 2)
        return loss

def bilateral_solver_local(reference: np.ndarray,
                           target: np.ndarray,
                           sigma_space: float = 32,
                           sigma_luma: float = 8,
                           lam: float = 32,
                           kernel_size: int = 21, 
                           ) -> np.ndarray:
    solver = BilateralSolverLocal(reference, target, sigma_space, sigma_luma, lam, kernel_size)
    solver.cuda()

    optimizer = torch.optim.Adam(solver.parameters(), lr=1e-3)
    for _ in tqdm(range(2000)): 
        optimizer.zero_grad()
        loss = solver()
        loss.backward()
        optimizer.step()
        print(loss)

    output = torch.Tensor(solver.output).view((target.shape[0], target.shape[1])).detach().cpu().numpy() * 255

    return output



# if __name__ == '__main__':
#     import cv2  
#     refer = np.array([
#         [1, 2, 3, 4, 5],
#         [6, 7, 8, 9, 10],
#         [11, 12, 13, 14, 15],
#         [16, 17, 18, 19, 20]
#         ], dtype=np.uint8)
#     tgt = np.array([
#         [1, 2, 3, 4, 5],
#         [6, 7, 8, 9, 10],
#         [11, 12, 13, 14, 15],
#         [16, 17, 18, 19, 20]
#         ], dtype=np.uint8)

#     out = bilateral_solver_local(refer, tgt, lam=1, sigma_space=32, sigma_luma=1)
#     cv2.imwrite('result.png', out)


if __name__ == '__main__':
    # 2d
    import cv2

    refer = cv2.imread('reference.png')
    tgt = cv2.imread('target.png', 0)

    out = bilateral_solver_local(refer, tgt, lam=1, sigma_space=32, sigma_luma=1)
    cv2.imwrite('result.png', out)

