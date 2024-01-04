import torch
import timm
from einops import rearrange, reduce

class MILModel(torch.nn.Module):
    """
    MIL Model w/ self attention.

    Source: https://arxiv.org/pdf/1703.03130.pdf
    """
    def __init__(self, backbone: str):
        super().__init__()
        self.backbone = timm.create_model(
            backbone,
            pretrained=True,
            num_classes=5,
            drop_path_rate=0.1,
        )
        self.backbone.set_grad_checkpointing()

        if "regnet" in backbone or \
           "maxvit" in backbone:
            num_features = self.backbone.head.fc.in_features
            self.backbone.head.fc = torch.nn.Identity()
            self.attention = torch.nn.Sequential(torch.nn.Linear(num_features, 2048), torch.nn.Tanh(), torch.nn.Linear(2048, 1))
            self.dropout = torch.nn.Dropout(p=0.5)
            self.fc = torch.nn.Linear(num_features, 5)
        else:
            self.backbone.classifier = torch.nn.Identity()
            self.attention = torch.nn.Sequential(torch.nn.Linear(self.backbone.num_features, 2048), torch.nn.Tanh(), torch.nn.Linear(2048, 1))
            self.dropout = torch.nn.Dropout(p=0.5)
            self.fc = torch.nn.Linear(self.backbone.num_features, 5)

    def forward(self, x, HAM_pct: float=0.0):
        b, tta, t = x.size()[:3]
        x = rearrange(x, "b tta t c h w -> (b tta t) c h w", b=b, t=t)
        x = self.backbone(x)
        x = self.dropout(x)
        x = rearrange(x, "(b tta t) f -> (b tta) t f", b=b, t=t, tta=tta)
        a = self.attention(x)

        # Randomly apply HAM to N out of the top K highest attention values HAM_pct of the time
        k, n = 4, 1
        if torch.rand(1).item() < HAM_pct:
            _, top_indices = a.topk(k, dim=1)
            mask = torch.ones_like(a)
            indices_to_mask = torch.randperm(k)[:n]
            mask.scatter_(1, top_indices[:, indices_to_mask], 0)
            a = a * mask

        a = torch.softmax(a, dim=1)
        x = torch.sum(x * a, dim=1)

        x = self.fc(x)
        x = rearrange(x, "(b tta) f -> b tta f", b=b, tta=tta)
        x = reduce(x, "b tta f -> b f", "mean") # TTA mean pooling
        return x
    
class StainNet(torch.nn.Module):
    """
    Model used for stain normalization. Not used
    in final submission.

    Source: https://github.com/khtao/StainNet
    """
    def __init__(self, input_nc=3, output_nc=3, n_layer=3, n_channel=32, kernel_size=1):
        super(StainNet, self).__init__()
        model_list = []
        model_list.append(torch.nn.Conv2d(input_nc, n_channel, kernel_size=kernel_size, bias=True, padding=kernel_size // 2))
        model_list.append(torch.nn.ReLU(True))
        for n in range(n_layer - 2):
            model_list.append(
                torch.nn.Conv2d(n_channel, n_channel, kernel_size=kernel_size, bias=True, padding=kernel_size // 2))
            model_list.append(torch.nn.ReLU(True))
        model_list.append(torch.nn.Conv2d(n_channel, output_nc, kernel_size=kernel_size, bias=True, padding=kernel_size // 2))

        self.rgb_trans = torch.nn.Sequential(*model_list)

    def forward(self, x):
        return self.rgb_trans(x)