from flask import Flask, render_template, request, jsonify, send_file
import os
import cv2
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
import base64
from PIL import Image
import io
import json

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
YOLO_MODEL_PATH = 'car_parts_detector.pt'
SAM_CHECKPOINT_PATH = 'SAM/sam_vit_h_4b8939.pth'
SAM_MODEL_TYPE = 'vit_h'
DEVICE = "cpu"

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize models
yolo_model = YOLO(YOLO_MODEL_PATH)
sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH)
sam.to(device=DEVICE)
predictor = SamPredictor(sam)

# Cache for SAM embeddings
image_embeddings_cache = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_colors(n):
    """Generate n distinct colors"""
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7
        value = 0.9
        rgb = plt.cm.hsv(hue)[:3]
        colors.append(rgb)
    return colors

def process_detection(image):
    """Run YOLO detection on image"""
    results = yolo_model(image)
    return results[0]

def create_segmentation_mask(image, box):
    """Create segmentation mask using SAM"""
    # Convert box to the format SAM expects
    box = np.array([box[0], box[1], box[2], box[3]])
    
    # Set image and get mask
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        box=box,
        multimask_output=False,
        point_coords=None,
        point_labels=None
    )
    
    return masks[0]

def image_to_base64(image_array):
    """Convert numpy array to base64 string"""
    img = Image.fromarray(image_array)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/app')
def main_app():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_parts():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400

        # Get the image file
        image_file = request.files['image']
        if not image_file.filename:
            return jsonify({'success': False, 'error': 'No image file selected'}), 400

        # Read the image
        image_data = image_file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'success': False, 'error': 'Failed to decode image'}), 400

        # Run YOLO detection
        results = yolo_model(image)
        
        # Process results
        detected_parts = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                # Get confidence
                conf = box.conf[0].cpu().numpy()
                # Get class
                cls = int(box.cls[0].cpu().numpy())
                
                if conf > 0.5:  # Confidence threshold
                    detected_parts.append({
                        'label': yolo_model.names[cls],
                        'confidence': float(conf),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)]
                    })

        # Draw bounding boxes on the image
        annotated_image = image.copy()
        for part in detected_parts:
            x1, y1, x2, y2 = part['bbox']
            # Use a bright color for the box (yellow)
            box_color = (0, 255, 255)  # BGR for yellow
            text_color = (0, 0, 0)     # Black text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.3  # Increased font size
            thickness = 4
            label = f"{part['label']} {part['confidence']:.2f}"
            # Draw bounding box
            cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), box_color, thickness)
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness=2)
            # Draw filled rectangle behind text
            cv2.rectangle(annotated_image, (int(x1), int(y1) - text_height - baseline - 10), (int(x1) + text_width + 10, int(y1)), box_color, -1)
            # Draw text (centered vertically in the box)
            cv2.putText(annotated_image, label, (int(x1) + 5, int(y1) - 8), font, font_scale, text_color, thickness=2, lineType=cv2.LINE_AA)

        # Convert the annotated image to base64
        _, buffer = cv2.imencode('.jpg', annotated_image)
        detected_image_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'success': True,
            'detected_image': detected_image_base64,
            'detected_parts': detected_parts
        })

    except Exception as e:
        print(f"Error in detect_parts: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/segment', methods=['POST'])
def segment():
    try:
        if 'image' not in request.files or 'parts' not in request.form:
            return jsonify({'success': False, 'error': 'Missing required data'}), 400

        # Get the image file
        image_file = request.files['image']
        if not image_file.filename:
            return jsonify({'success': False, 'error': 'No image file selected'}), 400

        # Read the image
        image_data = image_file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'success': False, 'error': 'Failed to decode image'}), 400

        # Get parts data
        try:
            parts = json.loads(request.form['parts'])
        except json.JSONDecodeError:
            return jsonify({'success': False, 'error': 'Invalid parts data'}), 400

        # Convert image to RGB for SAM
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Generate colors for each class
        unique_classes = list(set(part['label'] for part in parts))
        colors = generate_colors(len(unique_classes))
        color_map = {cls: color for cls, color in zip(unique_classes, colors)}
        
        # Create segmentation overlay
        overlay = image_rgb.copy()
        masks = {}
        
        # Process parts in batches to avoid memory issues
        batch_size = 2
        for i in range(0, len(parts), batch_size):
            batch = parts[i:i + batch_size]
            for part in batch:
                label = part['label']
                bbox = part['bbox']
                
                # Convert bbox to numpy array for SAM
                box = np.array(bbox)
                
                try:
                    # Get segmentation mask
                    mask = create_segmentation_mask(image_rgb, box)
                    masks[label] = mask
                    
                    # Apply colored overlay
                    color = color_map[label]
                    overlay[mask] = (overlay[mask] * 0.5 + np.array(color) * 255 * 0.5).astype(np.uint8)
                except Exception as e:
                    print(f"Error processing part {label}: {str(e)}")
                    continue
        
        # Convert the segmented image to base64
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        segmented_image_base64 = base64.b64encode(buffer).decode('utf-8')

        # Prepare color legend data
        color_legend = [{
            'label': cls,
            'color': [float(c) for c in color]
        } for cls, color in color_map.items()]

        return jsonify({
            'success': True,
            'segmented_image': segmented_image_base64,
            'colors': color_legend
        })

    except Exception as e:
        print(f"Error in segment: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/extract', methods=['POST'])
def extract():
    try:
        if 'image' not in request.files or 'part_index' not in request.form:
            return jsonify({'success': False, 'error': 'Missing required data'}), 400

        # Get the image file
        image_file = request.files['image']
        if not image_file.filename:
            return jsonify({'success': False, 'error': 'No image file selected'}), 400

        # Read the image
        image_data = image_file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'success': False, 'error': 'Failed to decode image'}), 400

        # Convert image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get part index
        try:
            part_index = int(request.form['part_index'])
        except (ValueError, TypeError):
            return jsonify({'success': False, 'error': 'Invalid part index'}), 400

        # Get the bounding box for the selected part
        box = np.array([
            float(request.form.get('bbox[0]', 0)),
            float(request.form.get('bbox[1]', 0)),
            float(request.form.get('bbox[2]', 0)),
            float(request.form.get('bbox[3]', 0))
        ])

        try:
            # Get segmentation mask
            mask = create_segmentation_mask(image_rgb, box)
            
            # Create extracted part image
            extracted = image_rgb.copy()
            extracted[~mask] = 255  # White background
            
            # Create masked original
            masked_original = image_rgb.copy()
            masked_original[mask] = 255  # Make extracted part white
            
            # Convert images to base64
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(extracted, cv2.COLOR_RGB2BGR))
            extracted_base64 = base64.b64encode(buffer).decode('utf-8')
            
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(masked_original, cv2.COLOR_RGB2BGR))
            masked_base64 = base64.b64encode(buffer).decode('utf-8')

            return jsonify({
                'success': True,
                'extracted_part': extracted_base64,
                'masked_original': masked_base64
            })

        except Exception as e:
            print(f"Error in mask creation: {str(e)}")
            return jsonify({'success': False, 'error': 'Failed to create mask'}), 500

    except Exception as e:
        print(f"Error in extract: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/replace', methods=['POST'])
def replace():
    try:
        if 'original_image' not in request.files or 'replacement_image' not in request.files:
            return jsonify({'success': False, 'error': 'Missing required images'}), 400

        # Get the original image
        original_file = request.files['original_image']
        if not original_file.filename:
            return jsonify({'success': False, 'error': 'No original image provided'}), 400

        # Get the replacement image
        replacement_file = request.files['replacement_image']
        if not replacement_file.filename:
            return jsonify({'success': False, 'error': 'No replacement image provided'}), 400

        # Get bounding box coordinates
        try:
            box = np.array([
                float(request.form.get('bbox[0]', 0)),
                float(request.form.get('bbox[1]', 0)),
                float(request.form.get('bbox[2]', 0)),
                float(request.form.get('bbox[3]', 0))
            ])
        except (ValueError, TypeError) as e:
            return jsonify({'success': False, 'error': f'Invalid bounding box data: {str(e)}'}), 400

        # Read the original image
        original_data = original_file.read()
        nparr = np.frombuffer(original_data, np.uint8)
        original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if original is None:
            return jsonify({'success': False, 'error': 'Failed to decode original image'}), 400
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

        # Read the replacement image
        replacement_data = replacement_file.read()
        nparr = np.frombuffer(replacement_data, np.uint8)
        replacement = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if replacement is None:
            return jsonify({'success': False, 'error': 'Failed to decode replacement image'}), 400
        replacement_rgb = cv2.cvtColor(replacement, cv2.COLOR_BGR2RGB)

        # Get the mask for the selected part
        try:
            mask = create_segmentation_mask(original_rgb, box)
        except Exception as e:
            print(f"Error creating mask: {str(e)}")
            return jsonify({'success': False, 'error': 'Failed to create mask for replacement'}), 500

        # Get bounding box of the mask
        y_indices, x_indices = np.where(mask)
        if len(y_indices) == 0 or len(x_indices) == 0:
            return jsonify({'success': False, 'error': 'Invalid mask area'}), 400
            
        x1, x2 = x_indices.min(), x_indices.max()
        y1, y2 = y_indices.min(), y_indices.max()

        try:
            # Resize replacement to match the mask area
            replacement_resized = cv2.resize(replacement_rgb, (x2 - x1 + 1, y2 - y1 + 1))
            
            # Create result image
            result = original_rgb.copy()
            mask_region = mask[y1:y2+1, x1:x2+1]
            result[y1:y2+1, x1:x2+1][mask_region] = replacement_resized[mask_region]

            # Convert result to base64
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            result_base64 = base64.b64encode(buffer).decode('utf-8')

            return jsonify({
                'success': True,
                'result_image': result_base64
            })

        except cv2.error as e:
            print(f"OpenCV error: {str(e)}")
            return jsonify({'success': False, 'error': 'Failed to process replacement image'}), 500
        except Exception as e:
            print(f"Error in image processing: {str(e)}")
            return jsonify({'success': False, 'error': str(e)}), 500

    except Exception as e:
        print(f"Error in replace: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)