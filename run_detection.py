import os
import json
import base64
import requests
from PIL import Image, ImageDraw, ImageFont

api_key = 'ST2dC0JPcQQ2wjaZm8Cm'
model_id = "microplastic_100/4"
api_url = "https://detect.roboflow.com"
image_path = "ocean.jpg"

print(f"Analyzing image: {image_path}")

with open(image_path, 'rb') as image_file:
    image_data = base64.b64encode(image_file.read()).decode('utf-8')

url = f"{api_url}/{model_id}?api_key={api_key}"
response = requests.post(url, json=image_data, headers={'Content-Type': 'application/json'})

if response.status_code == 200:
    result = response.json()
    predictions = result.get('predictions', [])
    
    print("\n" + "="*50)
    print(f"Total microplastics detected: {len(predictions)}")
    
    if predictions:
        print("\nDetailed Results:")
        for i, pred in enumerate(predictions, 1):
            confidence = pred.get('confidence', 0) * 100
            class_name = pred.get('class', 'unknown')
            print(f"  {i}. Class: {class_name} | Confidence: {confidence:.1f}%")
    else:
        print("No microplastics detected")
    print("="*50 + "\n")
    
    # Save JSON results
    with open('results.json', 'w') as f:
        json.dump(result, f, indent=2)
    print("✓ Results saved to results.json")
    
    # Draw bounding boxes on image
    if predictions:
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        
        # Try to use a better font, fall back to default if not available
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
            small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except:
            font = ImageFont.load_default()
            small_font = font
        
        # Color map for different classes
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
            
            # Get color for this class
            color = colors.get(class_name, '#FF6B9D')
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw label background
            label = f"{class_name} {confidence*100:.0f}%"
            bbox = draw.textbbox((x1, y1), label, font=small_font)
            draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=color)
            
            # Draw label text
            draw.text((x1, y1), label, fill='white', font=small_font)
        
        # Save annotated image
        output_path = image_path.replace('.', '_detected.')
        img.save(output_path)
        print(f"✓ Annotated image saved to {output_path}")
        
        # Display the image
        img.show()
        print("✓ Opening annotated image...")
    
else:
    print(f"Error: {response.status_code} - {response.text}")
