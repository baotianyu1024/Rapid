"""Dense benchmark shapes shared with the row-wise TorchRec comparison."""

import torch
from torch import nn


def mlp(input_width, widths):
    layers = []
    for index, width in enumerate(widths):
        layers.append(nn.Linear(input_width, width))
        if index + 1 < len(widths):
            layers.append(nn.ReLU())
        input_width = width
    return nn.Sequential(*layers)


class ReferenceCore(nn.Module):
    def __init__(self, model, dense_count, table_count, dim=32):
        super().__init__()
        self.model = model
        self.table_count = table_count
        self.dim = dim
        if model == 'dcnv2':
            width = table_count * dim + 128
            self.bottom = nn.Sequential(mlp(dense_count, [512, 256, 128]), nn.ReLU())
            self.cross_u = nn.ModuleList(nn.Linear(width, 512, bias=False) for _ in range(3))
            self.cross_v = nn.ModuleList(nn.Linear(512, width) for _ in range(3))
            self.top = mlp(width, [1024, 1024, 512, 256, 1])
        elif model == 'dcn':
            width = table_count * dim + dense_count
            self.cross_w = nn.ParameterList(nn.Parameter(torch.empty(width)) for _ in range(6))
            self.cross_b = nn.ParameterList(nn.Parameter(torch.zeros(width)) for _ in range(6))
            for weight in self.cross_w:
                nn.init.normal_(weight, std=0.01)
            self.deep = nn.Sequential(nn.Linear(width, 1024), nn.ReLU(), nn.Dropout(0.5),
                                      nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.5))
            self.top = nn.Linear(width + 1024, 1)
        elif model == 'dlrm':
            self.bottom = nn.Sequential(mlp(dense_count, [512, 256, dim]), nn.ReLU())
            interactions = (table_count + 1) * table_count // 2
            self.top = mlp(dim + interactions, [1024, 1024, 512, 256, 1])
        elif model in ('wide_deep', 'deepfm'):
            width = table_count * dim + dense_count
            self.deep = mlp(width, [1024, 1024, 1] if model == 'wide_deep'
                            else [400, 400, 400, 1])
            self.dense_linear = nn.Linear(dense_count, 1)
        else:
            raise ValueError(f'unsupported reference model: {model}')

    def forward(self, dense, embeddings):
        if self.model == 'dcnv2':
            x0 = torch.cat((*embeddings, self.bottom(dense)), dim=1)
            x = x0
            for u, v in zip(self.cross_u, self.cross_v):
                x = x + x0 * v(u(x))
            return self.top(x)
        if self.model == 'dcn':
            x0 = torch.cat((*embeddings, dense), dim=1)
            x = x0
            for weight, bias in zip(self.cross_w, self.cross_b):
                x = x + x0 * (x @ weight).unsqueeze(1) + bias
            return self.top(torch.cat((self.deep(x0), x), dim=1))
        if self.model == 'dlrm':
            bottom = self.bottom(dense)
            fields = torch.stack((bottom, *embeddings), dim=1)
            gram = torch.bmm(fields, fields.transpose(1, 2))
            left, right = torch.tril_indices(self.table_count + 1, self.table_count + 1,
                                             offset=-1, device=dense.device)
            return self.top(torch.cat((bottom, gram[:, left, right]), dim=1))

        fields = torch.stack([embedding[:, :self.dim] for embedding in embeddings], dim=1)
        features = torch.cat((fields.flatten(1), dense), dim=1)
        logits = (self.deep(features) +
                  torch.stack([embedding[:, self.dim] for embedding in embeddings], dim=1)
                  .sum(1, keepdim=True) + self.dense_linear(dense))
        if self.model == 'deepfm':
            fm = 0.5 * (fields.sum(dim=1).square() - fields.square().sum(dim=1))
            logits = logits + fm.sum(dim=1, keepdim=True)
        return logits
