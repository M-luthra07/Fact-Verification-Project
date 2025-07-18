from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import zipfile
import gdown

GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
app = Flask(__name__)
MODEL_DIR = "saved_model"
GDRIVE_FILE_ID = "1BrzkJwAIO0_G1V7hxNahkcAt6zxXSNKq"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

# Download and extract model only if not already present
if not os.path.exists(MODEL_DIR):
    print("Model not found. Downloading from Google Drive...")
    gdown.download(GDRIVE_URL, MODEL_ZIP, quiet=False)
    with zipfile.ZipFile(MODEL_ZIP, 'r') as zip_ref:
        zip_ref.extractall(MODEL_DIR)
    print("Model downloaded and extracted.")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
model.eval()

# ───────────────────── Routes ─────────────────────

@app.route('/')
def home():
    return render_template("index.html", prediction=None, input_text="")

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get("user_input", "").strip()
    if not text:
        return render_template("index.html", error="Please enter some text.", prediction=None, input_text="")

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    label = "True" if predicted_class == 1 else "False"
    return render_template("index.html", prediction=label, input_text=text)

@app.route('/analysis')
def analysis():
    metrics = {
        "epochs": [1, 2, 3, 4, 5],
        "train_loss": [0.8, 0.6, 0.4, 0.3, 0.2],
        "val_loss": [0.9, 0.7, 0.5, 0.35, 0.25],
        "train_acc": [0.6, 0.7, 0.8, 0.85, 0.9],
        "val_acc": [0.58, 0.68, 0.75, 0.82, 0.87],
        "true_count": 420,
        "false_count": 80
    }
    return render_template("analysis.html", **metrics)

@app.route('/api/metrics')
def metrics_api():
    return jsonify({
        "epochs": [1, 2, 3, 4, 5],
        "train_loss": [0.8, 0.6, 0.4, 0.3, 0.2],
        "val_loss": [0.9, 0.7, 0.5, 0.35, 0.25],
        "train_acc": [0.6, 0.7, 0.8, 0.85, 0.9],
        "val_acc": [0.58, 0.68, 0.75, 0.82, 0.87],
        "true_count": 420,
        "false_count": 80
    })

# ───────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, port=5000)
