from simple_cnn import SimpleCNN
import torch
from PIL import Image
from torchvision import transforms

# Configurar la transformación de la imagen
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Cargar el modelo
model = SimpleCNN()
model.load_state_dict(torch.load('simple_cnn.pth', weights_only=True))
model.eval()

def process_image(image_path):
    try:
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0)  # Añadir una dimensión para el batch
        with torch.no_grad():
            output = model(image)
        _, predicted = torch.max(output, 1)
        return predicted.item()
    except FileNotFoundError:
        print(f"El archivo de imagen en {image_path} no se encuentra.")
        return None

