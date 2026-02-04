import os
import json
import base64
import requests
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image, ImageDraw, ImageFont
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Roboflow API configuration
API_KEY = 'ST2dC0JPcQQ2wjaZm8Cm'
MODEL_ID = "microplastic_100/4"
API_URL = "https://detect.roboflow.com"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_microplastics(image_path):
    """Detect microplastics using Roboflow API"""
    with open(image_path, 'rb') as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')
    
    url = f"{API_URL}/{MODEL_ID}?api_key={API_KEY}"
    response = requests.post(url, json=image_data, headers={'Content-Type': 'application/json'})
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API Error: {response.status_code} - {response.text}")

def draw_predictions(image_path, predictions, output_path):
    """Draw bounding boxes on image"""
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    # Font setup
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        font = ImageFont.load_default()
        small_font = font
    
    # Color map
    colors = {
        'film': '#FF6B9D',
        'fragment': '#4ECDC4', 
        'pallet': '#95E1D3',
        'pellet': '#F38181',
        'fiber': '#AA96DA',
        'foam': '#FCBAD3'
    }
    
    for pred in predictions:
        x = pred['x']
        y = pred['y']
        width = pred['width']
        height = pred['height']
        confidence = pred['confidence']
        class_name = pred['class']
        
        # Calculate box coordinates
        x1 = x - width / 2
        y1 = y - height / 2
        x2 = x + width / 2
        y2 = y + height / 2
        
        color = colors.get(class_name, '#FF6B9D')
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        
        # Draw label
        label = f"{class_name} {confidence*100:.0f}%"
        bbox = draw.textbbox((x1, y1-25), label, font=small_font)
        draw.rectangle([bbox[0]-4, bbox[1]-2, bbox[2]+4, bbox[3]+2], fill=color)
        draw.text((x1, y1-25), label, fill='white', font=small_font)
    
    img.save(output_path)
    return output_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Detect microplastics
            result = detect_microplastics(filepath)
            predictions = result.get('predictions', [])
            
            # Draw boxes on image
            result_filename = f"detected_{filename}"
            result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
            draw_predictions(filepath, predictions, result_path)
            
            # Prepare response
            detection_summary = []
            for pred in predictions:
                detection_summary.append({
                    'class': pred['class'],
                    'confidence': round(pred['confidence'] * 100, 1)
                })
            
            return jsonify({
                'success': True,
                'total_detected': len(predictions),
                'detections': detection_summary,
                'result_image': f'/results/{result_filename}'
            })
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/results/<filename>')
def serve_result(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Simple ASCII-only startup banner to avoid Windows console encoding issues
    print("\n" + "=" * 60)
    print("Microplastic Detection App")
    print("=" * 60)
    print("Open your browser to: http://localhost:5001")
    print("=" * 60 + "\n")
    app.run(debug=True, host='127.0.0.1', port=5001, threaded=True)
