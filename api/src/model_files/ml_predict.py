from PIL import Image
import torch
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import pickle
import io
import json

# Model class to define the architecture
class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        self.conv1= nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5)
        self.conv2= nn.Conv2d(in_channels=6,out_channels=12,kernel_size=5)
        self.conv3= nn.Conv2d(in_channels=12,out_channels=24,kernel_size=5)
        self.conv4= nn.Conv2d(in_channels=24,out_channels=48,kernel_size=5)
        
        
        self.fc1 = nn.Linear(in_features=48*12*12,out_features=240)
        self.fc2 = nn.Linear(in_features=240,out_features=120)
        self.out = nn.Linear(in_features=120,out_features=17)
        
        
    def forward(self,t):
        t = t
        
        t=self.conv1(t)
        t=F.relu(t)
        t=F.max_pool2d(t,kernel_size = 2, stride = 2)
        
        
        t=self.conv2(t)
        t=F.relu(t)
        t=F.max_pool2d(t,kernel_size = 2, stride = 2)

        t=self.conv3(t)
        t=F.relu(t)
        t=F.max_pool2d(t,kernel_size = 2, stride = 2)

        t=self.conv4(t)
        t=F.relu(t)
        t=F.max_pool2d(t,kernel_size = 2, stride = 2)
        
        t=t.reshape(-1,48*12*12)
        t=self.fc1(t)
        t=F.relu(t)
        
        
        t=self.fc2(t)
        t=F.relu(t)
        
        t=self.out(t)
        
        
        return t


def get_remedy(plant_disease):
    with open("model_files/data.json", 'rb') as f:
	    remedies = json.load(f)
    # Get remedy for the given plant disease
    for key in remedies:
        if key == plant_disease:
            return(remedies[key])
        
def get_nutrition(plant_disease):
    with open("model_files/nutri.json", 'rb') as f:
        nutrient = json.load(f)
        for key in nutrient:
            if key == plant_disease:
                return(nutrient[key])


# to avoid gradients update
@torch.no_grad()
def predict_plant(model, imgdata):
    with open('model_files/labels_2.json', 'rb') as lb:
        labels = pickle.load(lb)

    loaded_model = model
    loaded_model.load_state_dict(torch.load("model_2.pth"))
    loaded_model.eval()

    # Converting Base64 string to Image
    image = Image.open(io.BytesIO(imgdata))
    # Resizing Image
    resize = transforms.Compose([transforms.Resize((256, 256))])
    image = ToTensor()(image)

    # Getting prediction from model
    y_result = model(resize(image).unsqueeze(0))
    result_idx = y_result.argmax(dim=1).item()  # Get the predicted class index

    # Find the corresponding plant disease or healthy label
    plant_disease = None
    for key, value in labels.items():
        if value == result_idx:
            plant_disease = key
            break

    if plant_disease is None:
        # If the label is not found, return an error response
        raise ValueError("Prediction result does not match any known label.")

    # Check if the plant is healthy
    if "healthy" in plant_disease.lower():
        remedy = "Plant is healthy, no disease detected."
        nutrients = None
    else:
        try:
            remedy = get_remedy(plant_disease)
        except:
            remedy = "Not Found!"

        try:
            nutrients = get_nutrition(plant_disease)
        except:
            nutrients = "Nutrients data not available."

    return plant_disease, remedy, nutrients
