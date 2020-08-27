#!/usr/bin/env python
# -*- coding-utf-8 -*-
# xuer ----time:
import torch
import torch.nn as nn

class focal_BCELoss(nn.Module):
    def __init__(self, alpha=10, gamma=2):
        super(focal_BCELoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target, eps=1e-7):
        input = torch.clamp(input, eps, 1 - eps)  # 避免出现log(0)的情况

        # loss = torch.zeros(input.shape[0])
        # loss = loss.cuda() if torch.cuda.is_available() else loss
        # target_loss = torch.zeros(input.shape[1])
        # target_loss = target_loss.cuda() if torch.cuda.is_available() else target_loss
        # for i in range(input.shape[0]):
        #     for j in range(input.shape[1]):
        #         # target_loss[j] = -torch.log(input[i, j]) + self.alpha * (1 - input[i, j]) ** self.gamma if target[i, j] == 1 else -(torch.log(1 - input[i, j]))
        #         target_loss[j] = -torch.log(input[i, j]) if target[i, j] == 1 else -(torch.log(1 - input[i, j]))
        #     loss[i] = torch.mean(target_loss)
        # final_loss = torch.mean(loss)

        # loss = -target * torch.log(input) * self.alpha * (1 - input) ** self.gamma - (1 - target) * torch.log(1 - input)

        # loss = -torch.sum(torch.mul(target, torch.log(input)), torch.mul((1 - target), torch.log(1 - input)))

        loss = -(target * torch.log(input)) * torch.exp(self.alpha * (1 - input) ** self.gamma) - (1 - target) * torch.log(1 - input)
        final_loss = torch.mean(loss)
        return final_loss

t = torch.Tensor([[1, 0, 0]])
p = torch.Tensor([[1, 0.1, 0.1]])
focalloss = focal_BCELoss()
loss = focalloss(p, t)
torchbce = nn.BCELoss()
torch_loss = torchbce(p, t)
print(loss)
print(torch_loss)

