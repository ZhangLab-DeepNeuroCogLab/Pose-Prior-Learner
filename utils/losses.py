import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F
import os

threshold = nn.Threshold(1, 0)

def compute_boundary_loss(core, double, single, img_size):
    """
    compute boundary loss, boundaries are 0 and 1
    loss = x if x smaller or greater than 0, 1
    0 otherwise
    """
    core = core.view(core.shape[0], -1, core.shape[3])
    single = single.view(single.shape[0], -1, single.shape[3])
    double = double.view(double.shape[0], -1, double.shape[3])

    comb = torch.cat([core, double, single], dim=1)

    # normalize to range -1  to 1
    comb = (comb / img_size) * 2 - 1

    return threshold(torch.abs(comb)).sum(1).mean()

def compute_template_boundary_loss(points):
    return threshold(torch.abs(points)).sum(1).mean()

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        os.environ['TORCH_HOME'] = os.path.abspath(os.getcwd())
        blocks = [torchvision.models.vgg16(weights='DEFAULT').features[:4].eval(),
                  torchvision.models.vgg16(weights='DEFAULT').features[4:9].eval(),
                  torchvision.models.vgg16(weights='DEFAULT').features[9:16].eval(),
                  torchvision.models.vgg16(weights='DEFAULT').features[16:23].eval()]
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        x = x * 0.5 + 0.5
        y = y * 0.5 + 0.5
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        x = F.interpolate(x, mode='bilinear', size=(224, 224), align_corners=False)
        y = F.interpolate(y, mode='bilinear', size=(224, 224), align_corners=False)
        perceptual_loss = 0.0

        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)

            perceptual_loss += torch.nn.functional.l1_loss(x, y)

        return perceptual_loss