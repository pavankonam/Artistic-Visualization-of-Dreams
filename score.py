import json
import torch
import torchvision.transforms as transforms
from PIL import Image
from azureml.core.model import Model

# Initialize model at deployment start
def init():
    global model
    model_path = Model.get_model_path("dl_dreams")  # Ensure model name matches registered model
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()

# Run inference when request is received
def run(data):
    try:
        input_data = json.loads(data)
        image_path = input_data["image_path"]

        # Preprocess image
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)

        # Run model
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)

        class_labels = ["Fear", "Uncertainty", "Calm", "Other"]
        return json.dumps({"prediction": class_labels[predicted.item()]})
    
    except Exception as e:
        return json.dumps({"error": str(e)})
