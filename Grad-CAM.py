import argparse
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from config import get_config
from networks.vision_transformer import SwinUnet
# from networks.Unet import AttU_Net
from scipy.ndimage import gaussian_filter

# 配置参数
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./data/Synapse/test_vol_h5',
                    help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--output_dir', default='./predict_FSA-Net_noFSAS/best_model.pth', type=str, help='output dir')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--volume_path', type=str, default="./data/Synapse",
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', default=True, action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='./predict', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cfg', type=str, default='./configs/swin_tiny_patch4_window7_224_lite.yaml', required=False, metavar="FILE", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

parser.add_argument("--n_class", default=9, type=int)
parser.add_argument("--split_name", default="test", help="Directory of the input list")
parser.add_argument("--pretrained", default="./pretrained_ckpt", help="Directory of the input list")
args = parser.parse_args()


# 自定义GradCAM类（适配SwinUnet）
class SwinUnetGradCAM:
    def __init__(self, model, target_layer_name="decoder4.blocks.3"):
        self.model = model.eval()
        self.target_layer_name = target_layer_name
        self.feature_maps = []
        self.gradients = []

        # 注册钩子
        target_layer = self._find_target_layer()
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _find_target_layer(self):
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                return module
        raise ValueError(f"未找到目标层: {self.target_layer_name}")

    def _forward_hook(self, module, input, output):
        self.feature_maps.append(output.detach())

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0].detach())

    def _preprocess_h5(self, h5_path, slice_idx=0):
        """预处理H5医学图像"""
        with h5py.File(h5_path, "r") as f:
            volume = f["image"][:]  # 假设H5文件存储3D数据

        # 提取切片并处理维度
        slice_data = volume[slice_idx]
        if slice_data.ndim == 3:  # 处理多通道情况
            slice_data = slice_data.transpose(2, 0, 1)  # [H, W, C] -> [C, H, W]

        # 转换为张量并标准化
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((args.img_size, args.img_size)),
            transforms.Normalize(mean=[0.485], std=[0.229])  # 医学图像常用参数
        ])
        return transform(slice_data).unsqueeze(0).cuda()  # 单通道处理

    def generate(self, h5_path, slice_idx=0, class_idx=None):
        # 数据预处理
        input_tensor = self._preprocess_h5(h5_path, slice_idx)

        return input_tensor

        # 前向传播
        self.model.zero_grad()
        output = self.model(input_tensor)  # [1, C, H, W]

        # 设置目标类
        if class_idx is None:
            class_idx = torch.argmax(output.mean(dim=(2, 3)), dim=1).item()

        # 反向传播
        target = output[:, class_idx, :, :].mean()
        target.backward()

        # 计算CAM
        features = self.feature_maps[-1]
        grads = self.gradients[-1]

        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = (weights * features).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # 后处理
        cam = F.interpolate(cam, size=input_tensor.shape[2:],
                            mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


# 可视化函数
def visualize(cam, h5_path, slice_idx, alpha=0.4, sigma=1.5, cmap='viridis'):
    """优化后的可视化函数（像素感更弱）"""
    with h5py.File(h5_path, "r") as f:
        raw_slice = f["image"][slice_idx]  # 原始切片

    # 对CAM进行高斯模糊和平滑处理
    cam_smooth = gaussian_filter(cam, sigma=sigma)

    # 创建自定义透明度渐变颜色映射
    cmap = plt.get_cmap(cmap)
    cmap._init()
    cmap._lut[:, -1] = np.linspace(0, 0.8, 259)  # 透明度渐变

    plt.figure(figsize=(12, 6), dpi=100)  # 提高DPI

    # 原始切片（增强对比度）
    plt.subplot(1, 2, 1)
    plt.imshow(raw_slice,
               cmap='gray',
               vmin=np.percentile(raw_slice, 5),  # 动态对比度调整
               vmax=np.percentile(raw_slice, 95),
               interpolation='bicubic')
    plt.title("Original Slice")
    plt.axis('off')

    # 叠加可视化（优化叠加方式）
    plt.subplot(1, 2, 2)
    plt.imshow(raw_slice,
               cmap='gray',
               vmin=np.percentile(raw_slice, 5),
               vmax=np.percentile(raw_slice, 95),
               interpolation='bicubic')

    # 使用自定义颜色映射和透明度
    heatmap = plt.imshow(cam_smooth,
                         cmap=cmap,
                         alpha=alpha,  # 全局透明度控制
                         interpolation='hanning')  # 更平滑的插值方式

    plt.title("Smooth Grad-CAM Activation")
    plt.axis('off')

    # 添加颜色条说明强度
    cbar = plt.colorbar(heatmap, fraction=0.046, pad=0.04)
    cbar.set_label('Activation Intensity')

    plt.tight_layout()
    plt.show()

class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()


import urllib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt
from numpy.ma.core import clip


class Hook:
    """Attaches to a module and records its activations and gradients."""

    def __init__(self, module: nn.Module):
        self.data = None
        self.hook = module.register_forward_hook(self.save_grad)

    def save_grad(self, module, input, output):
        self.data = output
        output.requires_grad_(True)
        output.retain_grad()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()

    @property
    def activation(self) -> torch.Tensor:
        return self.data

    @property
    def gradient(self) -> torch.Tensor:
        return self.data.grad


def gradCAM(
        model: nn.Module,
        input: torch.Tensor,
        target: torch.Tensor,
        layer: nn.Module
) -> torch.Tensor:
    # 梯度归0
    if input.grad is not None:
        input.grad.data.zero_()

    #
    requires_grad = {}
    for name, param in model.named_parameters():
        requires_grad[name] = param.requires_grad
        param.requires_grad_(False)

    # 添加钩子函数
    assert isinstance(layer, nn.Module)
    with Hook(layer) as hook:
        # 前向和后向传播
        output = model(input)
        target = output[:, 1, :, :].mean()
        target.backward()

        grad = hook.gradient.float()
        act = hook.activation.float()

        # 在空间维度进行平均池化来得到权重
        alpha = grad.mean(dim=(2, 3), keepdim=True)
        # 通道维度加权求和
        gradcam = torch.sum(act * alpha, dim=1, keepdim=True)
        # 去除负值，只想要正值
        gradcam = torch.clamp(gradcam, min=0)

    # resize
    gradcam = F.interpolate(
        gradcam,
        input.shape[2:],
        mode='bicubic',
        align_corners=False)

    # 存储梯度设置
    for name, param in model.named_parameters():
        param.requires_grad_(requires_grad[name])

    return gradcam


def normalize(x: np.ndarray) -> np.ndarray:
    # Normalize to [0, 1].
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x


# Modified from: https://github.com/salesforce/ALBEF/blob/main/visualization.ipynb
def getAttMap(img, attn_map, blur=True):
    if blur:
        attn_map = gaussian_filter(attn_map, 0.02 * max(img.shape[:2]))
    attn_map11 = normalize(attn_map)

    # 生成3通道热力图（jet颜色）
    cmap = plt.get_cmap('jet')
    attn_map_c = cmap(attn_map11)[:, :, :3]  # 保留RGB通道，删除Alpha通道（原代码np.delete可替换为此）

    # 调整attn_map为3通道（与attn_map_c通道数一致）
    attn_base = (1 - attn_map11 ** 0.7).reshape(attn_map11.shape + (1,))  # 单通道
    attn_map = np.repeat(attn_base, 3, axis=-1)  # 复制为3通道：(224,224,3)

    # 计算热力图部分（已为3通道）
    attn_map1 = (attn_map11 ** 0.7).reshape(attn_map11.shape + (1,)) * attn_map_c  # 自动广播为3通道

    # 相加（此时两者均为3通道）
    attn_map += attn_map1
    return attn_map

def viz_attn(img, attn_map, blur=True, alpha=0.5):  # alpha 控制透明度
    _, ax = plt.subplots(1, 1, figsize=(5, 5))

    # 绘制原图
    ax.imshow(img)

    # 生成带颜色的注意力图并叠加
    attn_overlay = getAttMap(img, attn_map, blur)  # 假设 getAttMap 返回3通道数组
    ax.imshow(attn_overlay, alpha=alpha)  # alpha 控制注意力图透明度

    ax.axis("off")
    # plt.show()
    plt.savefig('noFSAS.png')
    plt.show()


def load_image(img_path, resize=None):
    image = Image.open(img_path).convert("RGB")
    if resize is not None:
        image = image.resize((resize, resize))
    return np.asarray(image).astype(np.float32) / 255.

if __name__ == "__main__":
    # 初始化模型
    config = get_config(args)
    model = SwinUnet(config, img_size=args.img_size, num_classes=args.num_classes).cuda()

    # 加载权重
    state_dict = torch.load(args.output_dir, weights_only=True)
    model.load_state_dict(state_dict)

    # # 创建GradCAM实例
    gradcam = SwinUnetGradCAM(model.swin_unet, target_layer_name="output")
    #
    # # 生成示例结果
    # h5_file = "./data/Synapse/test_vol_h5/case0004.npy.h5"
    # input_tensor = gradcam.generate(h5_file, slice_idx=50)
    #
    # target_layers = 'output'
    # with GradCAM(model=model,
    #              target_layers=target_layers) as cam:
    #     grayscale_cam = cam(input_tensor=input_tensor,
    #                         targets=0)

    # image_url = './data/Synapse/test_vol_h5/case0022.npy.h5'

    # image_caption = 'the cat'
    # clip_model = "RN50"  # ["RN50", "RN101", "RN50x4", "RN50x16"]
    saliency_layer = "output"  # ["layer4", "layer3", "layer2", "layer1"]
    blur = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess = clip.load(clip_model, device=device, jit=False)

    h5_file = "./data/Synapse/test_vol_h5/case0032.npy.h5"
    input_tensor = gradcam.generate(h5_file, slice_idx=105)
    # image_np = load_image(image_path, model.visual.input_resolution)
    # text_input = clip.tokenize([image_caption]).to(device)

    # 计算热力图
    attn_map = gradCAM(
        model,
        input_tensor,
        input_tensor,
        getattr(model.swin_unet, saliency_layer)
    )
    attn_map = attn_map.squeeze().detach().cpu().numpy()

    viz_attn(input_tensor.squeeze(0).squeeze(0).cpu().numpy(), attn_map, blur)
    # 可视化
    # visualize(cam, h5_file, slice_idx=60)