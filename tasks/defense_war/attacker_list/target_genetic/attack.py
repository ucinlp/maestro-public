from typing import List, Iterator, Dict, Tuple, Any, Type

import numpy as np
import torch
from copy import deepcopy

np.random.seed(1901)

class Attack:
    def __init__(
        self,
        vm, device, attack_path,
        min_val = 0,
        max_val = 1,
        image_size = [1, 3, 32, 32],
        n_population = 50,
        n_generation = 50,
        mask_rate = 0.2,
        temperature = 0.1,
        use_mask = False,
        step_size = 0.2,
        child_rate = 0.4,
        mutate_rate = 0.8,
        l2_threshold = 12
    ):
        """
        args:
            vm: virtual model is wrapper used to get outputs/gradients of a model.
            device: system on which code is running "cpu"/"cuda"
            min_val: minimum value of each element in original image
            max_val: maximum value of each element in original image
                     each element in perturbed image should be in the range of min_val and max_val
            attack_path: Any other sources files that you may want to use like models should be available in ./submissions/ folder and loaded by attack.py.
                         Server doesn't load any external files. Do submit those files along with attack.py
        """
        self.vm = vm
        self.device = device
        self.attack_path = attack_path
        self.min_val = 0
        self.max_val = 1
        self.image_size = image_size
        self.n_population = n_population
        self.n_generation = n_generation
        self.mask_rate = mask_rate
        self.temperature = temperature
        self.use_mask = use_mask
        self.step_size = step_size
        self.child_rate = child_rate
        self.mutate_rate = mutate_rate
        self.l2_threshold = l2_threshold
    def attack(
        self, original_images: torch.tensor, labels: torch.tensor, target_label = None,
    ):
        assert torch.is_tensor(original_images), "original_images should be a torch tensor."
        assert torch.is_tensor(labels), "labels should be a torch tensor."

        original_images = original_images.to(self.device)
        labels = labels.to(self.device)
        target_labels = target_label * torch.ones_like(labels).to(self.device)
        perturbed_image = original_images.detach().cpu().numpy()
        self.mask = np.random.binomial(1, self.mask_rate, size=self.image_size).astype(
            "bool"
        )

        self.original_image = original_images.detach().cpu().numpy()
        #print("o i:", np.shape(self.original_image))
        population = self.init_population(self.original_image)
        examples = [(labels[0], labels[0], np.squeeze(x)) for x in population[:10]]
        success = False
        for g in range(self.n_generation):
            # print("generation: ", g)
            population, output, scores, best_index, detect_outputs = self.eval_population(
                population, target_label
            )
            if (np.argmax(output[best_index, :]) == target_label)& (detect_outputs[best_index].item() == 0):
                # print(f"Attack Success!")
                perturbed_image = np.array(population[best_index])
                success = True
                break
            # print(type(population))
        perturbed_image = np.array(population[best_index])
        assert type(perturbed_image) == np.ndarray, "perturbed_image should be numpy"
        perturbed_image = torch.FloatTensor(perturbed_image)
        image_tensor = perturbed_image.to(self.device)

        adv_outputs, _ = self.vm.get_batch_output(image_tensor)
        final_pred = adv_outputs.max(1, keepdim=True)[1]
        correct = 0
        correct += (final_pred == target_labels).sum().item()
        # print("out", g)
        return perturbed_image.detach().cpu().numpy(), correct

    def init_population(self, original_image: np.ndarray):
        """
        Initialize the population to n_population of images. Make sure to perturbe each image.
        args:
            original_image: image to be attacked
        return:
            a list of perturbed images initialized from orignal_image
        """
        return np.array(
            [self.perturb(original_image[0]) for _ in range(self.n_population)]
        )

    def eval_population(self, population, target_label):
        """
        evaluate the population, pick the parents, and then crossover to get the next
        population
        args:
            population: current population, a list of images
            target_label: target label we want the imageto be classiied, int
        return:
            population: population of all the images
            output: output of the model
            scores: the "fitness" of the image, measured as logits of the target label
            best_indx: index of the best image in the population
        """
        output, scores, detect_outputs = self.fitness(population, target_label)
        logits = np.exp(scores / self.temperature)
        select_probs = logits / np.sum(logits)

        score_ranks = np.argsort(scores)[::-1]
        best_index = score_ranks[0]
        # print(detect_outputs.shape, output.shape, best_index)
        if (np.argmax(output[best_index, :]) == target_label) & (detect_outputs[best_index].item() == 0):
            return population, output, scores, best_index, detect_outputs

        # the elite gene that's defeintely in the next population without perturbation
        elite = [population[best_index]]

        # strong and fit genes passed down to next generation, they have a chance to mutate
        survival_number = int((1 - self.child_rate) * (self.n_population - 1))
        survived = [population[idx] for idx in score_ranks[1 : survival_number + 1]]
        survived = [
            self.perturb(x) if np.random.uniform() < self.mutate_rate else x
            for x in survived
        ]

        # offsprings of strong genes
        child_number = self.n_population - 1 - survival_number
        mom_index = np.random.choice(self.n_population, child_number, p=select_probs)
        dad_index = np.random.choice(self.n_population, child_number, p=select_probs)
        childs = [
            self.crossover(population[mom_index[i]], population[dad_index[i]])
            for i in range(child_number)
        ]
        # childs = [self.perturb(childs[i]) for i in range(len(childs))]
        population = np.array(elite + survived + childs)
        return population, output, scores, best_index, detect_outputs
    def crossover(self, x1, x2):
        """
        crossover two images to get a new one. We use a uniform distribution with p=0.5
        args:
            x1: image #1
            x2: image #2
        return:
            x_new: newly crossovered image
        """
        x_new = x1.copy()
        for i in range(x1.shape[1]):
            for j in range(x1.shape[2]):
                for k in range(x1.shape[3]):
                    if np.random.uniform() < 0.5:
                        x_new[0][i][j][k] = x2[0][i][j][k]
        return x_new


    def softmax(self, x):
        after_softmax = []
        for single_x in x:
            r=np.exp(single_x - np.max(single_x))
            p = r / np.sum(r)
            after_softmax.append(p)
        after_softmax = np.array(after_softmax)
        return after_softmax

    def fitness(self, image: np.ndarray, target: int):
        """
        evaluate how fit the current image is
        return:
            output: output of the model
            scores: the "fitness" of the image, measured as logits of the target label
        """
        output, detect_outputs = self._get_batch_outputs_numpy(image)
        softmax_output = self.softmax(output)
        scores = softmax_output[:, target]
        return output, scores, detect_outputs

    def perturb(self, image):
        """
        perturb a single image with some constraints and a mask
        args:
            image: the image to be perturbed
        return:
            perturbed: perturbed image
        """
        if not self.use_mask:
            adv_images = image + np.random.randn(*self.mask.shape) * self.step_size
            # perturbed = np.maximum(np.minimum(adv_images,self.original_image+0.5), self.original_image-0.5)
            delta = np.expand_dims(adv_images - self.original_image, axis=0)
            # Assume x and adv_images are batched tensors where the first dimension is
            # a batch dimension
            eps = self.l2_threshold
            mask = (
                np.linalg.norm(delta.reshape((delta.shape[0], -1)), ord=2, axis=1)
                <= eps
            )
            scaling_factor = np.linalg.norm(
                delta.reshape((delta.shape[0], -1)), ord=2, axis=1
            )
            scaling_factor[mask] = eps
            delta *= eps / scaling_factor.reshape((-1, 1, 1, 1))
            perturbed = self.original_image + delta
            perturbed = np.squeeze(np.clip(perturbed, 0, 1), axis=0)
        else:
            perturbed = np.clip(
                image + self.mask * np.random.randn(*self.mask.shape) * self.step_size,
                0,
                1,
            )
        return perturbed

    def _get_batch_outputs_numpy(self, image: np.ndarray):
        #print("image:", np.shape(image))
        image = np.array([i[0] for i in image])
        image_tensor = torch.FloatTensor(image)
        image_tensor = image_tensor.to(self.device)
        outputs, detect_outputs = self.vm.get_batch_output(image_tensor)
        return outputs.detach().cpu().numpy(), detect_outputs
