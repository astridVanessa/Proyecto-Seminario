import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from flask import Flask, request, render_template
from PIL import Image
import os

app = Flask(__name__)

# Definir la clase SimpleCNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64*32*32, 128)  # Asumiendo imágenes 128x128
        self.fc2 = nn.Linear(128, len(train_dataset.classes))  # Número de clases

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 64*32*32)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Configuración de dispositivos y carga del modelo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
model.load_state_dict(torch.load('modelo_dibujos_psicologicos.pth'))
model.eval()

# Transformación para la imagen de entrada
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def predict(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
            prediction = predict(file_path)
            return render_template('result.html', prediction=prediction)
    return render_template('index.html')

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
