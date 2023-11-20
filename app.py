from flask import Flask, request, render_template
import os
import torch
import io
import ast
from torchvision import models, transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to load labels from the file
def load_labels(label_file):
    with open(label_file, "r") as file:
        data = file.read()
        labels = ast.literal_eval(data)
    return labels

# Load the labels
IMAGENET_LABELS = load_labels("./ImageNet_labels/lookup.txt")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load a pre-trained PyTorch model
def load_model():
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()
    return model

# Preprocess the image and predict the class
def detect_objects(image_file):
    model = load_model()

    # Define the image transformations
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Open the image from memory, convert to RGB, and apply transformations
    img = Image.open(io.BytesIO(image_file.read())).convert("RGB")
    img = preprocess(img)
    img = img.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(img)

    _, predicted = torch.max(outputs, 1)
    predicted_class = predicted[0].item()

    class_label = IMAGENET_LABELS.get(predicted_class, f"Unknown class ID: {predicted_class}")

    print(f"Predicted Class: {class_label}")

    return {"class": class_label}

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            results = detect_objects(file)
            return render_template('display.html', results=results)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 5000, debug=True)