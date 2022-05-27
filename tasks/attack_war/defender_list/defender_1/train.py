"""
The template for the students to train the model.
Please do not change the name of the functions in Adv_Training.
"""
import sys
import time

from torchvision.transforms import transforms

sys.path.append("../../../")
import torch
import random
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from utils import get_dataset
import importlib.util
from predict import DLA

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class VirtualModel:
    def __init__(self, device, model) -> None:
        self.device = device
        self.model = model

    def get_batch_output(self, images):
        predictions = []
        # for image in images:
        predictions = self.model(self.model.preprocess(images)).to(self.device)  # self.model.preprocess(images)
            # predictions.append(prediction)
        # predictions = torch.tensor(predictions)
        return predictions

    def get_batch_input_gradient(self, original_images, labels):
        original_images.requires_grad = True
        self.model.eval()
        outputs = self.model(original_images)
        loss = F.nll_loss(outputs, labels)
        self.model.zero_grad()
        loss.backward()
        data_grad = original_images.grad.data
        return data_grad


class Adv_Training():
    """
    The class is used to set the defense related to adversarial training and adjust the loss function. Please design your own training methods and add some adversarial examples for training.
    The perturb function is used to generate the adversarial examples for training.
    """
    def __init__(self, device, file_path, target_label=None, epsilon=0.3, min_val=0, max_val=1):
        self.num_classes = 4
        sys.path.append(file_path)
        from predict import LeNet
        self.model = DLA(num_classes=self.num_classes).to(device)
        self.epsilon = epsilon
        self.device = device
        self.min_val = min_val
        self.max_val = max_val
        self.target_label = target_label
        self.perturb = self.load_perturb("../attacker_list/nontarget_FGSM")
        self.target_labels = range(self.num_classes)

        self.trapdoor_ratio = 0.1
        self.patterns_per_label = 1
        self.pattern_size = 3
        self.mask_ratio = 0.1
        self.pattern_color = [1.0, 1.0, 1.0]
        self.num_clusters = 7
        self.img_shape = (3, 32, 32)

        self.trapdoor_image_pattern_dict = self.craft_trapdoors()  # create and store trapdoor perturbation patterns

    def train(self, trainset, valset, device, epoches=200):

        # self.model.to(device)
        self.model.train()
        trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=128, num_workers=1)

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters())
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoches)

        best_accuracy = 0.0

        # inserting the trapdoors
        print("START Train Model")
        for epoch in range(epoches):  # loop over the dataset multiple times

            self.model.train()
            train_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                print('\rEpoch: %d,  Progress %3d%%' % (epoch, 100 * (batch_idx*128/len(trainset))), end="")
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = transform_train(inputs).to(device)

                for i in range(inputs.shape[0]):
                    if np.random.uniform(0.0, 1.0) < self.trapdoor_ratio:
                        # make this a trapdoor image to lure attackers to by changing its label and using its pattern
                        inputs[i] = self.generate_trapdoor_image(inputs[i], random.choice(range(self.num_classes)))

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss, 100. * correct / total, correct, total))
            print('\rEpoch: %d,  Progress %3d%%' % (epoch, 100))

            valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=True, num_workers=1)
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in valloader:
                    inputs = inputs.to(device)
                    inputs = transform_test(inputs).to(device)

                    labels = labels.to(device)
                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = correct / total

            print("Accuracy of the network on the val images: %.3f%%, best accuracy: %.3f%%" %
                  (100 * accuracy, 100 * best_accuracy)
                  )
            if accuracy > best_accuracy:
                print("Saving new best model...", end="")
                best_accuracy = accuracy
                torch.save(self.model.state_dict(), "defense_project-model.pth")
                print("Done")

            scheduler.step()
        return

    def generate_trapdoor_image(self, image, target_label: int):
        assert image.shape == self.img_shape
        assert np.max(image) <= 1.0, f"{np.max(image)}"
        mask, pattern = random.choice(self.trapdoor_image_pattern_dict[target_label])
        mask = np.copy(mask)

        assert np.max(pattern) <= 1.0, f"{np.max(pattern)}"
        assert np.max(mask) <= 1.0, f"{np.max(mask)}"


        adv_img = mask * pattern + (1 - mask) * image

        return adv_img, target_label

    def craft_trapdoors(self):

        total_ls = {}

        for y_target in self.target_labels:
            cur_pattern_ls = []

            for _ in range(self.patterns_per_label):
                tot_mask = np.zeros(self.img_shape)
                tot_pattern = np.zeros(self.img_shape)

                for p in range(self.num_clusters):
                    mask, _ = self.construct_mask_random_location()
                    tot_mask += mask

                    vs = np.random.uniform(0.0, 1.0, 6)

                    r = np.random.normal(vs[0], vs[1], self.img_shape[1:])
                    g = np.random.normal(vs[2], vs[3], self.img_shape[1:])
                    b = np.random.normal(vs[4], vs[5], self.img_shape[1:])
                    cur_pattern = np.stack([r, g, b], axis=0)
                    cur_pattern = cur_pattern * (mask != 0)
                    cur_pattern = np.clip(cur_pattern, 0, 1.0)
                    tot_pattern += cur_pattern

                tot_mask = (tot_mask > 0) * self.mask_ratio
                tot_pattern = np.clip(tot_pattern, 0, 1.0)
                assert np.max(tot_mask) <= 1.0, f"{np.max(tot_mask)}"
                assert np.max(tot_pattern) <= 1.0, f"{np.max(tot_pattern)}"
                cur_pattern_ls.append([tot_mask, tot_pattern])

            total_ls[y_target] = cur_pattern_ls

        return total_ls

    def construct_mask_random_location(self):

        c_col = random.choice(range(0, self.img_shape[1] - self.pattern_size + 1))
        c_row = random.choice(range(0, self.img_shape[0] - self.pattern_size + 1))

        mask = np.zeros(self.img_shape)
        pattern = np.zeros(self.img_shape)

        mask[:, c_row:c_row + self.pattern_size, c_col:c_col + self.pattern_size] = 1
        pattern[:, c_row:c_row + self.pattern_size, c_col:c_col + self.pattern_size] = self.pattern_color

        return mask, pattern

    def load_perturb(self, attack_path):
        spec = importlib.util.spec_from_file_location('attack', attack_path + '/attack.py')
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
        # for attack methods evaluator, the Attack class name should be fixed
        attacker = foo.Attack(VirtualModel(self.device, self.model), self.device, attack_path)  # TODO use PGD and FGSM attacks
        return attacker

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adv_training = Adv_Training(device, file_path='.')
    dataset_configs = {
                "name": "CIFAR10",
                "binary": True,
                "dataset_path": "../datasets/CIFAR10/student/",
                "student_train_number": 10000,
                "student_val_number": 1000,
                "student_test_number": 100,
    }

    dataset = get_dataset(dataset_configs)
    trainset = dataset['train']
    valset = dataset['val']
    testset = dataset['test']
    adv_training.train(trainset, valset, device)
    torch.save(adv_training.model.state_dict(), "defense_project-model.pth")


if __name__ == "__main__":
    main()
