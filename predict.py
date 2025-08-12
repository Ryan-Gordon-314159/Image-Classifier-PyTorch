import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from main import SimpleCNN  # Import the same model class

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same preprocessing as training
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load model
model = SimpleCNN(num_classes=10).to(device)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# CIFAR-10 class names
classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    img_t = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_t)
        _, predicted = torch.max(outputs, 1)
    
    return classes[predicted.item()]

if __name__ == "__main__":
    img_path = "path/to/your/image.jpg"
    prediction = predict_image(img_path)
    print(f"Predicted class: {prediction}")
