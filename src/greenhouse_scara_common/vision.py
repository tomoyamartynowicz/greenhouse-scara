"""Independent RGB/depth encoders; no unsynchronized spatial augmentations."""
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from .data import image_specs


def normalize_rgb(image):
    mean = image.new_tensor([.485, .456, .406])[None, :, None, None]
    std = image.new_tensor([.229, .224, .225])[None, :, None, None]
    return (image - mean) / std


def adapt_conv(backbone, channels):
    if channels == 3: return
    old = backbone.conv1
    conv = nn.Conv2d(channels, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
    with torch.no_grad():
        conv.weight.copy_(old.weight.mean(1, keepdim=True).repeat(1, channels, 1, 1) * 3 / channels)
    backbone.conv1 = conv


class MultiModalObsEncoder(nn.Module):
    def __init__(self, shape_meta, pretrained=False):
        super().__init__()
        self.specs = image_specs(shape_meta)
        self.models = nn.ModuleDict()
        self.lowdim = {k: tuple(v['shape']) for k, v in shape_meta['obs'].items() if v.get('type', 'low_dim') == 'low_dim'}
        for key, spec in self.specs.items():
            model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
            adapt_conv(model, spec['shape'][0])
            self._group_norm(model)
            model.fc = nn.Identity()
            self.models[key] = model

    @staticmethod
    def _group_norm(module):
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                setattr(module, name, nn.GroupNorm(min(32, child.num_features), child.num_features))
            else:
                MultiModalObsEncoder._group_norm(child)

    def output_shape(self):
        return (512 * len(self.specs) + sum(int(torch.tensor(shape).prod()) for shape in self.lowdim.values()),)

    def forward(self, obs):
        parts = []
        for key, spec in self.specs.items():
            image = obs[key]
            if tuple(image.shape[1:]) != tuple(spec['shape']):
                raise ValueError(f'{key}: got {tuple(image.shape)}, expected B,{spec["shape"]}')
            if spec['type'] == 'rgb': image = normalize_rgb(image)
            parts.append(self.models[key](image))
        parts.extend(obs[key].flatten(1) for key in self.lowdim)
        return torch.cat(parts, dim=-1)
