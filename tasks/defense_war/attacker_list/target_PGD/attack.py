from typing import List, Iterator, Dict, Tuple, Any, Type
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import DataLoader
import torch.nn.functional as F

class Attack:
    # PGD Attack
    def __init__(self, vm, device, attack_path, epsilons=[1,2,4,8,16], alpha=0.1, min_val=0, max_val=1, max_iters=50,  _type='l2'):
        # self.model = model._to(device)
        self.vm = vm
        self.device = device
        self.attack_path = attack_path
        self.epsilons = epsilons
        self.min_val = min_val
        self.max_val = max_val
        self.alpha = alpha
        self.max_iters = max_iters
        self._type = _type

    def project(self, x, original_x, epsilon, _type='l2'):
        if _type == 'linf':
            max_x = original_x + epsilon
            min_x = original_x - epsilon
            x = torch.max(torch.min(x, max_x), min_x)
        elif _type == 'l2':
            dist = (x - original_x)
            dist = dist.view(x.shape[0], -1)
            dist_norm = torch.norm(dist, dim=1, keepdim=True)
            mask = (dist_norm > epsilon).unsqueeze(2).unsqueeze(3)
            # dist = F.normalize(dist, p=2, dim=1)
            dist = dist / dist_norm
            dist *= epsilon
            dist = dist.view(x.shape)
            x = (original_x + dist) * mask.float() + x * (1 - mask.float())
        else:
            raise NotImplementedError
        return x

    def attack(self, original_images, labels, target_label = None, reduction4loss='mean', random_start=True):
        assert len(labels) == 1, "image and label size should be 1. Otherwise, consider attack_batch function."
        assert torch.is_tensor(original_images), "original_images should be a torch tensor."
        assert torch.is_tensor(labels), "labels should be a torch tensor."
        assert target_label != None, "target label should not be None"
        for epsilon in self.epsilons:
            self.epsilon = epsilon
            original_images = original_images.to(self.device)
            labels = labels.to(self.device)
            target_labels = target_label * torch.ones_like(labels).to(self.device)

            if random_start:
                rand_perturb = torch.FloatTensor(original_images.shape).uniform_(
                    -self.epsilon, self.epsilon)
                rand_perturb = rand_perturb.to(self.device)
                x = original_images + rand_perturb
                x.clamp_(self.min_val, self.max_val)
            else:
                x = original_images.clone()

            with torch.enable_grad():
                for _iter in range(self.max_iters):
                    grads = self.vm.get_batch_input_gradient(x.data, target_labels)
                    x.data -= self.alpha * torch.sign(grads.data)
                    x = self.project(x, original_images, self.epsilon, self._type)
                    x.clamp_(self.min_val, self.max_val)
                    outputs, detect_outputs = self.vm.get_batch_output(x)

                    if detect_outputs == [1]:
                        continue
                    else:
                        final_pred = outputs.max(1, keepdim=True)[1]
                        correct = (final_pred == target_labels).sum().item()
                        if correct == 1:
                            return x.cpu().detach().numpy(), correct
            if detect_outputs.item() == 1:
                continue
            else:
                final_pred = outputs.max(1, keepdim=True)[1]
                correct = (final_pred == target_labels).sum().item()
                if correct == 1:
                    return x.cpu().detach().numpy(), correct

        return x.cpu().detach().numpy(), 0

    def attack_batch(self, original_images, labels, target_label = None, reduction4loss='mean', random_start=True):
        assert torch.is_tensor(labels), "labels should be a torch tensor."
        for epsilon in self.epsilons:
            self.epsilon = epsilon
            original_images = original_images.to(self.device)
            labels = labels.to(self.device)
            target_labels = target_label * torch.ones_like(labels).to(self.device)

            if random_start:
                rand_perturb = torch.FloatTensor(original_images.shape).uniform_(
                    -self.epsilon, self.epsilon)
                rand_perturb = rand_perturb.to(self.device)
                x = original_images + rand_perturb
                x.clamp_(self.min_val, self.max_val)
            else:
                x = original_images.clone()

            with torch.enable_grad():
                for _iter in range(self.max_iters):
                    grads = self.vm.get_batch_input_gradient(x.data, target_labels)
                    x.data -= self.alpha * torch.sign(grads.data)
                    x = self.project(x, original_images, self.epsilon, self._type)
                    x.clamp_(self.min_val, self.max_val)
                    outputs, detect_outputs = self.vm.get_batch_output(x)
        return x.cpu().detach().numpy(), 0
