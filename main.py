from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
from torchvision import transforms
import io
import torch.nn as nn
from torchvision import datasets, transforms

train_data_dir = './archive/train'
train_dataset = datasets.ImageFolder(train_data_dir)

# Definiere die Transformations, die auf das hochgeladene Bild angewendet werden sollen
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Lade das trainierte Modell
class CNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 18 * 18, 128), 
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = CNN(num_classes=100)  # Hier wurde die Anzahl der Klassen korrigiert
model.load_state_dict(torch.load('sports_classification_model.pth'))
model.eval()

# Definiere die Klassenbezeichnungen
class_names = train_dataset.classes # Hier musst du die 100 Klassenbezeichnungen einfügen 

# Erstelle die FastAPI-App
app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Lade das Bild
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Wende die Transformationen an
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # Füge eine Batch-Dimension hinzu

    # Führe die Inferenz durch
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output.data, 1)
        predicted_class = class_names[predicted.item()]

    # Gib das Ergebnis zurück
    return {"predicted_class": predicted_class}