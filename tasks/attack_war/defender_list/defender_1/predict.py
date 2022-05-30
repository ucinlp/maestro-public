"""
The template for the students to predict the result.
Please do not change LeNet, the name of batch_predict and predict function of the Prediction.

"""
from PIL import Image
import glob
import os
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import scipy.ndimage

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        #self.batch1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        #self.batch2 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(10, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        #x = self.batch1(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = self.batch2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class Prediction():
    """
    The Prediction class is used for evaluator to load the model and detect or classify the images. The output of the batch_predict function will be checked which is the label.
    If the label is the same as the ground truth, it means you predict the image successfully. If the label is -1 and the image is an adversarial examples, it means you predict the image successfully. Other situations will be decided as failure.
    You can use the preprocess function to clean or check the input data are benign or adversarial for later prediction.
    """
    def __init__(self, device, file_path):
        self.device = device
        self.model = self.constructor(file_path).to(device)

    def constructor(self, file_path=None):
        model = LeNet()
        if file_path != None:
            model.load_state_dict(torch.load(file_path+'/defense_project-model.pth', map_location=self.device))
        model.eval()
        return model

    def preprocess(self, original_images):
        requires_grad=original_images.requires_grad
        original_images.requires_grad = False
        #actually changing the image
        #flip left to right
        perturbed_images = torch.flip(original_images,[3])
        #apply jpeg compression
        #this doesn't really do much, probably worsens the results
        #for i in range(len(perturbed_images)):
            #img = T.ToPILImage()(perturbed_images[i])
            #img.save("temp.jpeg", "jpeg",optimize=True, quality=45) #10,45 looked good
            #img=Image.open("temp.jpeg")
            #result=T.ToTensor()(img).to(self.device)
            #perturbed_images[i] = result
        #gaussian blur
        for i in range(len(perturbed_images)):
            temp=perturbed_images[i].cpu()
            for c in range(len(temp)):
                nparr=np.zeros_like(temp[c])
                nparr=scipy.ndimage.gaussian_filter(temp[c],5,mode='mirror') #4.5,5 work better
                temp[c]=T.ToTensor()(nparr).to(self.device)
            perturbed_images[i]=temp


        perturbed_images.requires_grad = requires_grad
        return perturbed_images

    def get_batch_output(self, images, with_preprocess=True, skip_detect=False):
        predictions = []
        # for image in images:
        predictions = self.model(images).to(self.device)
            # predictions.append(prediction)
        # predictions = torch.tensor(predictions)
        return predictions, [0]*images.shape[0]

    def get_batch_input_gradient(self, original_images, labels, lossf=None):
        original_images=self.preprocess(original_images)
        original_images.requires_grad = True
        self.model.eval()
        outputs = self.model(original_images)
        if lossf is None:
            loss = F.nll_loss(outputs, labels)
        else:
            loss = lossf(outputs, labels)
        self.model.zero_grad()
        loss.backward()
        data_grad = original_images.grad.data
        return data_grad

