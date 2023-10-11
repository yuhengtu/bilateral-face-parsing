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

        self.image_size = (reference.shape[1], reference.shape[0])

        position_x = torch.linspace(0, self.image_size[0], self.image_size[0])[None, :].repeat((self.image_size[1], 1))
        position_y = torch.linspace(0, self.image_size[1], self.image_size[1])[:, None].repeat((1, self.image_size[0]))
        position_xy = torch.stack([position_x, position_y], dim=-1)

        position_xy = position_xy.view((-1, 2))

        p_ij = - torch.sum(
            (position_xy[:, None, :] - position_xy[None, :, :]) ** 2, dim=-1, dtype=torch.float32
        ) / (2 * sigma_space ** 2)
        del position_xy

        guide = torch.tensor(reference, dtype=torch.float32).view((-1,))
        
        g_ij = - (guide[:, None] - guide[None, :]) ** 2 / (2 * sigma_luma ** 2)
        del guide

        w_ij = torch.exp(
            p_ij + g_ij
        )
        del p_ij
        del g_ij

        self.w_ij = nn.Parameter(w_ij, requires_grad=False)
        del w_ij
        self.target = nn.Parameter(torch.Tensor(target / 255.).view((-1, 1)), requires_grad=False)
        self.output = nn.Parameter(torch.Tensor(target / 255.).view((-1, 1)), requires_grad=True)
        self.lam = lam

    def forward(self) -> Tensor:
        loss = self.image_size[0] * self.image_size[1] * self.lam * \
               torch.mean(self.w_ij * (self.output.T - self.output) ** 2) \
               + torch.mean((self.output - self.target) ** 2)
        return loss
    
class BilateralSolverLocal(nn.Module):
    def __init__(self,
                 reference: np.ndarray,
                 target: np.ndarray,
                 sigma_space: float = 32,
                 sigma_luma: float = 8,
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

        # 创建卷积层，用于处理位置信息
        self.conv = nn.Conv2d(
            1, kernel_size * kernel_size - 1, kernel_size, padding=(kernel_size - 1) // 2, padding_mode='replicate'
        )
        self.conv.weight = nn.Parameter(weight, requires_grad=False)

        # 获取图像大小
        self.image_size = (reference.shape[1], reference.shape[0])

        # 创建x和y方向上的位置信息
        position_x = torch.linspace(
            0, self.image_size[0], self.image_size[0]
        )[None, :].repeat((self.image_size[1], 1))[None, None, :, :]
        position_y = torch.linspace(
            0, self.image_size[1], self.image_size[1]
        )[:, None].repeat((1, self.image_size[0]))[None, None, :, :]

        # 使用卷积层处理位置信息，将位置信息变换成相似性矩阵
        position_x_ij = self.conv_ij(position_x)
        position_y_ij = self.conv_ij(position_y)
        position_ij = - (position_x_ij ** 2 + position_y_ij ** 2) / (2 * sigma_space ** 2)

        # 初始化参考图像的相似性矩阵
        reference_ij = 0
        if len(reference.shape) == len(target.shape):
            reference = reference[:, :, None]
        for c in range(reference.shape[-1]):
            # 使用卷积层处理每个通道的参考图像，将其变换成相似性矩阵
            reference_c_ij = self.conv_ij(torch.Tensor(reference[:, :, c])[None, None, :, :])
            reference_ij -= reference_c_ij ** 2 / (2 * sigma_luma ** 2)

        # 计算双边滤波器权重
        self.w_ij = nn.Parameter(torch.exp(position_ij + reference_ij), requires_grad=False)
        self.target = nn.Parameter(torch.Tensor(target / 255.), requires_grad=False)
        self.output = nn.Parameter(torch.Tensor(target / 255.), requires_grad=True)
        self.lam = lam

    def conv_ij(self, inp: Tensor) -> Tensor:
        batch_size = inp.shape[0]
        out = self.conv(inp)
        return out.view((batch_size, -1))

    def forward(self) -> Tensor:
        # 计算损失函数
        loss = self.image_size[0] * self.image_size[1] * self.lam * torch.mean(
            self.w_ij * self.conv_ij(self.output[None, None, :, :]) ** 2
        ) + torch.mean((self.output - self.target) ** 2)
        return loss


class BilateralSolverLocal3D(nn.Module):
    def __init__(self,
                 reference: np.ndarray,
                 target: np.ndarray,
                 sigma_space: float = 32,
                 sigma_luma: float = 8,
                 lam: float = 128,
                 space_kernel_size: int = 21,
                 time_kernel_size: int = 5,
                 ) -> None:
        super().__init__()
        weight = torch.zeros((
            space_kernel_size * space_kernel_size * time_kernel_size - 1, 1,
            time_kernel_size, space_kernel_size, space_kernel_size
        ))
        num = 0
        for i in range(space_kernel_size):
            for j in range(space_kernel_size):
                for k in range(time_kernel_size):
                    if i == j == (space_kernel_size - 1) // 2 and k == (time_kernel_size - 1) // 2:
                        continue
                    weight[num, 0, k, i, j] = -1
                    weight[
                        num, 0, (time_kernel_size - 1) // 2, (space_kernel_size - 1) // 2, (space_kernel_size - 1) // 2
                    ] = 1
                    num += 1

        self.conv = nn.Conv3d(
            1, space_kernel_size * space_kernel_size * time_kernel_size - 1,
            (time_kernel_size, space_kernel_size, space_kernel_size),
            padding=((time_kernel_size - 1) // 2, (space_kernel_size - 1) // 2, (space_kernel_size - 1) // 2),
            padding_mode='replicate'
        )
        self.conv.weight = nn.Parameter(weight, requires_grad=False)

        self.image_size = (reference.shape[0], reference.shape[2], reference.shape[1])

        position_x = torch.linspace(
            0, self.image_size[1], self.image_size[1]
        )[None, None, :].repeat((self.image_size[0], self.image_size[2], 1))[None, None, :, :, :]
        position_y = torch.linspace(
            0, self.image_size[2], self.image_size[2]
        )[None, :, None].repeat((self.image_size[0], 1, self.image_size[1]))[None, None, :, :, :]
        position_z = torch.linspace(
            0, self.image_size[0], self.image_size[0]
        )[:, None, None].repeat((1, self.image_size[2], self.image_size[1]))[None, None, :, :, :]
        position_x_ij = self.conv_ij(position_x)
        position_y_ij = self.conv_ij(position_y)
        position_z_ij = self.conv_ij(position_z)
        position_ij = - (position_x_ij ** 2 + position_y_ij ** 2 + position_z_ij ** 2) / (2 * sigma_space ** 2)

        reference_ij = 0
        if len(reference.shape) == len(target.shape):
            reference = reference[:, :, :, None]
        for c in range(reference.shape[-1]):
            reference_c_ij = self.conv_ij(torch.Tensor(reference[:, :, :, c])[None, None, :, :, :])
            reference_ij -= reference_c_ij ** 2 / (2 * sigma_luma ** 2)

        self.w_ij = nn.Parameter(torch.exp(position_ij + reference_ij), requires_grad=False)
        self.target = nn.Parameter(torch.Tensor(target / 255.), requires_grad=False)
        self.output = nn.Parameter(torch.Tensor(target / 255.), requires_grad=True)
        self.lam = lam

    def conv_ij(self, inp: Tensor) -> Tensor:
        batch_size = inp.shape[0]
        out = self.conv(inp)
        return out.view((batch_size, -1))

    def forward(self) -> Tensor:
        loss = self.image_size[0] * self.image_size[1] * self.image_size[2] * self.lam * torch.mean(
            self.w_ij * self.conv_ij(self.output[None, None, :, :, :]) ** 2
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


def bilateral_solver_local_3d(reference: np.ndarray,
                              target: np.ndarray,
                              sigma_space: float = 32,
                              sigma_luma: float = 8,
                              lam: float = 32,
                              space_kernel_size: int = 21,
                              time_kernel_size: int = 5,
                           ) -> np.ndarray:
    solver = BilateralSolverLocal3D(
        reference, target, sigma_space, sigma_luma, lam, space_kernel_size, time_kernel_size
    )
    solver.cuda()
    optimizer = torch.optim.Adam(solver.parameters(), lr=1e-3)
    for _ in tqdm(range(1000)):
        optimizer.zero_grad()
        loss = solver()
        loss.backward()
        optimizer.step()
        print(loss)

    output = torch.Tensor(solver.output).view(
        (target.shape[0], target.shape[1], target.shape[2])
    ).detach().cpu().numpy() * 255

    return output


if __name__ == '__main__':
    # 2d
    import cv2

    refer = cv2.imread('reference.png')
    tgt = cv2.imread('target.png', 0)

    out = bilateral_solver_local(refer, tgt, lam=1, sigma_space=32, sigma_luma=1)
    cv2.imwrite('result.png', out)

