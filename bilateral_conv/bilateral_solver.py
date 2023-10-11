import torch
from torch import nn, Tensor
import numpy as np
from tqdm import tqdm

class BilateralSolver(nn.Module):
    def __init__(self,
                 reference: np.ndarray,
                 target: np.ndarray,
                 sigma_space: float = 32,
                 sigma_luma: float = 8,
                 lam: float = 128,
                 ) -> None:
        super().__init__()
        
        # 从参考图像中获取图像大小
        self.image_size = (reference.shape[1], reference.shape[0])
        # 生成2D位置网格（x, y）
        position_x = torch.linspace(0, self.image_size[0]-1, self.image_size[0])[None, :].repeat((self.image_size[1], 1))
        position_y = torch.linspace(0, self.image_size[1]-1, self.image_size[1])[:, None].repeat((1, self.image_size[0]))
        position_xy = torch.stack([position_x, position_y], dim=-1)
        #position_xy每一行都是图形上的一个坐标
        position_xy = position_xy.view((-1, 2))
        # 计算位置之间的两两距离并创建空间相似性矩阵,公式第一项

        # (position_xy[:, None, :] - position_xy[None, :, :]) 8，8，2
        # 8,8,0表示i和j像素的x坐标差值
        # 8,8,1表示i和j像素的y坐标差值
        p_ij = - torch.sum(
            (position_xy[:, None, :] - position_xy[None, :, :]) ** 2, dim=-1, dtype=torch.float32
        ) / (2 * sigma_space ** 2)
        # sum dim=-1在第三维度上对平方差值进行求和
        # [8, 8] 每个元素表示两个像素点之间的 (x, y) 平方距离的总和。
        # p_ij[i, j] 表示了第 i 个像素点和第 j 个像素点之间的空间相似性分数
        del position_xy

        # 将参考图像转换为张量并展平成一维
        guide = torch.tensor(reference, dtype=torch.float32).view((-1,))
        # 计算亮度之间的两两差异并创建亮度相似性矩阵
        g_ij = - (guide[:, None] - guide[None, :]) ** 2 / (2 * sigma_luma ** 2)
        del guide

        # 通过结合空间和亮度相似性计算双边滤波器权重
        w_ij = torch.exp(
            p_ij + g_ij
        )
        del p_ij
        del g_ij

        # 创建双边滤波器权重的参数
        # self保证是固定参数，是buffer而不是优化参数parameter（保证to cpugpu，to float时和其他参数一致性）
        self.w_ij = nn.Parameter(w_ij, requires_grad=False)
        del w_ij
        
        # 创建目标图像的参数（归一化到[0, 1]），？（-1）行1列，buffer
        self.target = nn.Parameter(torch.Tensor(target / 255.).view((-1, 1)), requires_grad=False)
        print(self.target.shape) #[8, 1]
        print(f'self.target: {self.target}')
        # 创建具有梯度的输出图像的参数（归一化到[0, 1]），parameter
        self.output = nn.Parameter(torch.Tensor(target / 255.).view((-1, 1)), requires_grad=True)
        print(self.target.shape) #[8, 1]
        print(f'self.output: {self.output}')
        # 正则化参数
        self.lam = lam

    def forward(self) -> Tensor:
        # pytorch前向传播过程中计算双边损失
        # 就是优化公式
        # self.output.T - self.output，得8*8，维度一致（广播）
        # 公式第一项是对ij求sum，第二项对i求sum，mean函数要除以一个值，量纲匹配

        loss = self.image_size[0] * self.image_size[1] * self.lam * \
               torch.mean(self.w_ij * (self.output.T - self.output) ** 2) \
               + torch.mean((self.output - self.target) ** 2)
        print(f'loss_size:{loss.size}')
        return loss


def bilateral_solver(reference: np.ndarray,
                           target: np.ndarray,
                           sigma_space: float = 32,
                           sigma_luma: float = 8,
                           lam: float = 32,
                           ) -> np.ndarray:
    solver = BilateralSolver(reference, target, sigma_space, sigma_luma, lam)
    solver.cpu()
    optimizer = torch.optim.Adam(solver.parameters(), lr=1e-3)
    for _ in tqdm(range(1)):#2000
        optimizer.zero_grad()
        loss = solver()
        loss.backward()
        optimizer.step()
        print(loss)

    output = torch.Tensor(solver.output).view((target.shape[0], target.shape[1])).detach().cpu().numpy() * 255
    # 转换为PyTorch张量;view变为一个2D张量;detach从计算图中分离张量;移动到CPU;张量转换为NumPy数组; [0, 1]映射到[0, 255]变为标准的灰度图像
    return output


if __name__ == '__main__':
    # 2d
    import cv2
    gray_image = np.array([
        [100, 150, 200, 225],
        [50, 75, 125, 225],
    ], dtype=np.uint8)

    refer = gray_image
    tgt = gray_image
    out = bilateral_solver(refer, tgt, lam=1, sigma_space=32, sigma_luma=1)
    cv2.imwrite('result.png', out)

# if __name__ == '__main__':
#     # 2d
#     import cv2

#     refer = cv2.imread('reference.png')
#     tgt = cv2.imread('target.png', 0)

#     out = bilateral_solver_local(refer, tgt, lam=1, sigma_space=32, sigma_luma=1)
#     cv2.imwrite('result.png', out)

