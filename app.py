from flask import Flask, request, render_template
import os
import torch
import ast
from torchvision import models, transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
    # Use the new weights parameter to load the pre-trained model
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()
    return model

# Preprocess the image and predict the class
def detect_objects(image_path):
    model = load_model()

    # Define the image transformations
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Open the image, convert to RGB, and apply transformations
    img = Image.open(image_path).convert("RGB")
    img = preprocess(img)
    img = img.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(img)

    # Get the predicted class ID
    _, predicted = torch.max(outputs, 1)
    predicted_class = predicted[0].item()

    # Map the class ID to a human-readable label
    class_label = IMAGENET_LABELS.get(predicted_class, f"Unknown class ID: {predicted_class}")

    print(f"Predicted Class: {class_label}")

    return {"class": class_label}

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            results = detect_objects(file_path)
            return render_template('display.html', results=results)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
