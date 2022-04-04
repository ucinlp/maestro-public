"""
Author: c0ldstudy
2022-03-29 13:44:53
"""

import os
import torch
from torchvision import datasets, transforms
import numpy as np
import collections
import random

class TorchVisionDataset:
    """
    - name: the dataset name
    - subset: the subset of the main dataset. Dataset will be loaded as ``nlp.load_dataset(name, subset)``.
    - label_map: Mapping if output labels should be re-mapped. Useful
      if model was trained with a different label arrangement than
      provided in the ``nlp`` version of the dataset.
    - output_scale_factor (float): Factor to divide ground-truth outputs by.
        Generally, TextAttack goal functions require model outputs
        between 0 and 1. Some datasets test the model's correlation
        with ground-truth output, instead of its accuracy, so these
        outputs may be scaled arbitrarily.
    - shuffle (bool): Whether to shuffle the dataset on load.
    """

    def __init__(
        self, name,data, split="train", shuffle=False,
    ):
        self._name = name
        self._split = split
        self._dataset = data

        # Input/output column order, like (('premise', 'hypothesis'), 'label')

        self.input_columns, self.output_column = ("image", "label")

        self._i = 0
        self.examples = list(self._dataset)

        if shuffle:
            random.shuffle(self.examples)

    def __len__(self):
        return len(self._dataset)

    def _format_raw_example(self, raw_example):
        return raw_example

    def __next__(self):
        if self._i >= len(self.examples):
            raise StopIteration
        raw_example = self.examples[self._i]
        self._i += 1
        return self._format_raw_example(raw_example)

    def __getitem__(self, i):
        if isinstance(i, int):
            return self._format_raw_example(self.examples[i])
        else:
            # `i` could be a slice or an integer. if it's a slice,
            # return the formatted version of the proper slice of the list
            return [self._format_raw_example(ex) for ex in self.examples[i]]

    def get_json_data(self):
        if self.examples:
            new_data = []
            for idx, instance in enumerate(self.examples):
                new_instance = {}
                new_instance["image"] = instance[0].numpy().tolist()
                new_instance["label"] = instance[1]
                new_instance["uid"] = idx
                new_data.append(new_instance)
        return new_data

def get_dataset(name, dataset_configs):
    if name == "MNIST":
        return _read_mnist_dataset(dataset_configs, name)
    elif name == "CIFAR10":
        return _read_cifar10_dataset(dataset_configs, name)

def _split_by_labels(num_classes, train_data, server_number_sampled, train_server_path):
    subset_indices = []
    for i in range(num_classes):
        indices_xi = (torch.LongTensor(train_data.targets) == i).nonzero(as_tuple=True)[
            0
        ]
        sampled_indices = np.random.choice(
            indices_xi, server_number_sampled, replace=False
        )
        subset_indices.extend(sampled_indices)
    train_server_subset = torch.utils.data.Subset(train_data, subset_indices)
    torch.save(train_server_subset, train_server_path)
    return train_server_subset


def _read_mnist_dataset(dataset_configs, dataset_name):
    path = dataset_configs["dataset_path"]
    
    # 2.1 Training Data for Student
    train_path = os.path.join(path, "train_split.pt")
    if os.path.exists(train_path):
        train_student_subset = torch.load(train_path)
    else:
        train_data = datasets.MNIST(
            root=path,
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(),]),
        )
        num_classes = len(train_data.classes)
        student_number_sampled = dataset_configs["student_train_number"] // num_classes
        train_student_subset = _split_by_labels(
            num_classes, train_data, student_number_sampled, train_path
        )
    train_student_data = TorchVisionDataset(
        name=dataset_name, data=train_student_subset, split="train",
    )

    # 2.2 Test Data for Student
    test_path = os.path.join(path, "test_split.pt")
    if os.path.exists(test_path):
        test_student_subset = torch.load(test_path)
    else:
        test_data = datasets.MNIST(
            root=path,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(),]),
        )
        num_classes = len(test_data.classes)
        student_number_sampled = dataset_configs["student_test_number"] // num_classes
        test_student_subset = _split_by_labels(
            num_classes, test_data, student_number_sampled, test_path
        )
    test_student_data = TorchVisionDataset(
        name=dataset_name, data=test_student_subset, split="test",
    )
    print(f"train_student_data length: {len(train_student_data)}, test_student_data length: {len(test_student_data)}")
    return {
        "train": train_student_data,
        "test": test_student_data,
    }  # test_server_data,test_student_data




def _read_cifar10_dataset(dataset_configs, dataset_name, server = False):
    path = dataset_configs["dataset_path"]
    if server == True:
    # 1.1 Training Data for Server
        train_server_path = os.path.join(path, "train_server_split.pt")
        if os.path.exists(train_server_path):
            train_server_subset = torch.load(train_server_path)
        else:
            train_data = datasets.CIFAR10(
                root=path,
                train=True,
                download=True,
                transform=transforms.Compose([transforms.ToTensor()]),
            )
            num_classes = len(train_data.classes)
            server_number_sampled = dataset_configs["server_train_number"] // num_classes
            train_server_subset = _split_by_labels(
                num_classes, train_data, server_number_sampled, train_server_path
            )
        train_server_data = TorchVisionDataset(
            name=dataset_name, data=train_server_subset, split="train",
        )
        # 2.1 Test Data for Server
        test_server_path = os.path.join(path, "test_server_split.pt")
        if os.path.exists(test_server_path):
            test_server_subset = torch.load(test_server_path)
        else:
            test_data = datasets.CIFAR10(
                root=path,
                train=False,
                download=True,
                transform=transforms.Compose([transforms.ToTensor()]),
            )
            num_classes = len(test_data.classes)
            server_number_sampled = dataset_configs["server_test_number"] // num_classes
            test_server_subset = _split_by_labels(
                num_classes, test_data, server_number_sampled, test_server_path
            )
        test_server_data = TorchVisionDataset(
            name=dataset_name, data=test_server_subset, split="test",
        )



    # 1.2 Training Data for Student
    train_student_path = os.path.join(path, "train_student_split.pt")
    if False:
    # if os.path.exists(train_student_path):
        train_student_subset = torch.load(train_student_path)
    else:
        train_data = datasets.CIFAR10(
            root=path,
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        )

        num_classes = len(train_data.classes)
        student_number_sampled = dataset_configs["student_train_number"] // num_classes
        train_student_subset = _split_by_labels(
            num_classes, train_data, student_number_sampled, train_student_path
        )
    # print(len(train_student_subset))
    #     # print(train_data.max)
    #     # print(train_data.min)
    # print(torch.max(((train_student_subset[0][0]))))
    # print(torch.min(((train_student_subset[0][0]))))
    # exit()
    train_student_data = TorchVisionDataset(
        name=dataset_name, data=train_student_subset, split="train",
    )

    # 1.3 Validation Data for Student
    val_student_path = os.path.join(path, "val_student_split.pt")
    if os.path.exists(val_student_path):
        val_student_subset = torch.load(val_student_path)
    else:
        val_data = datasets.CIFAR10(
            root=path,
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        )
        num_classes = len(val_data.classes)
        student_number_sampled = dataset_configs["student_val_number"] // num_classes
        val_student_subset = _split_by_labels(
            num_classes, val_data, student_number_sampled, val_student_path
        )
    val_student_data = TorchVisionDataset(
        name=dataset_name, data=val_student_subset, split="val",
    )



    # 2.2 Test Data for Student
    test_student_path = os.path.join(path, "test_student_split.pt")
    if os.path.exists(test_student_path):
        test_student_subset = torch.load(test_student_path)
    else:
        test_data = datasets.CIFAR10(
            root=path,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        )
        num_classes = len(test_data.classes)
        student_number_sampled = dataset_configs["student_test_number"] // num_classes
        test_student_subset = _split_by_labels(
            num_classes, test_data, student_number_sampled, test_student_path
        )
    test_student_data = TorchVisionDataset(
        name=dataset_name, data=test_student_subset, split="test",
    )

    if server == True:
        print(
            f"train_server_data length: {len(train_server_data)}, train_student_data length: {len(train_student_data)}, test_server_data length: {len(test_server_data)}, val_student_data length: {len(val_student_data)}, test_student_data length: {len(test_student_data)}"
        )

        return {
            "train": train_student_data,
            "val": val_student_data,
            "test": test_student_data,
            "server_test": test_server_data
        }  # test_server_data,test_student_data
    else:
        print(
            f"train_student_data length: {len(train_student_data)}, val_student_data length: {len(val_student_data)}, test_student_data length: {len(test_student_data)}"
        )
        return {
            "train": train_student_data,
            "val": val_student_data,
            "test": test_student_data,
        }
