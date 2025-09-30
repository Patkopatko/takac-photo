#!/usr/bin/env python3
"""
Web interface for TAKAC PHOTO analyzer
Flask app with drag & drop upload
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import json
import tempfile
from werkzeug.utils import secure_filename
from professional_analyzer import ProfessionalImageAnalyzer
from vision_api_manager import VisionAPIManager
import hashlib
from datetime import datetime

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'heic', 'heif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page with upload interface"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and analysis"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Analyze with professional analyzer
            analyzer = ProfessionalImageAnalyzer()
            result = analyzer.analyze_image(filepath)

            # Add vision analysis if available
            vision_manager = VisionAPIManager()
            vision_result = vision_manager.analyze_image(filepath)

            if vision_result.success:
                result['services']['vision-enhanced'] = {
                    'status': 'success',
                    'result': {
                        'provider': vision_result.provider,
                        'scene_type': vision_result.scene_type,
                        'people_count': vision_result.people_count,
                        'face_count': vision_result.face_count,
                        'objects': vision_result.objects,
                        'text_detected': vision_result.text_detected,
                        'description': vision_result.description
                    },
                    'metrics': {
                        'processing_time_ms': 100,
                        'service_version': '1.0.0'
                    },
                    'error': None
                }

            return jsonify(result)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

        finally:
            # Clean up
            if os.path.exists(filepath):
                os.remove(filepath)

    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/providers')
def get_providers():
    """Get available vision providers"""
    manager = VisionAPIManager()
    return jsonify({
        'providers': manager.get_available_providers()
    })

@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

if __name__ == '__main__':
    # Create templates directory
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)

    # Create HTML template
    with open('templates/index.html', 'w') as f:
        f.write('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TAKAC PHOTO - Image Analyzer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            color: white;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
        }
        .upload-area {
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            margin-bottom: 30px;
        }
        #drop-zone {
            border: 3px dashed #ccc;
            border-radius: 10px;
            padding: 60px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
        }
        #drop-zone:hover, #drop-zone.drag-over {
            border-color: #667eea;
            background: #f8f9ff;
        }
        #drop-zone p {
            font-size: 1.2em;
            color: #666;
            margin-bottom: 10px;
        }
        #file-input {
            display: none;
        }
        .preview {
            margin-top: 20px;
            display: none;
        }
        .preview img {
            max-width: 300px;
            max-height: 300px;
            border-radius: 10px;
        }
        .results {
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            display: none;
        }
        .result-section {
            margin-bottom: 20px;
        }
        .result-section h3 {
            color: #667eea;
            margin-bottom: 10px;
        }
        .json-view {
            background: #f5f5f5;
            border-radius: 10px;
            padding: 20px;
            overflow-x: auto;
            max-height: 500px;
            overflow-y: auto;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .metadata-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .metadata-item {
            background: #f8f9ff;
            padding: 15px;
            border-radius: 8px;
        }
        .metadata-label {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }
        .metadata-value {
            font-size: 1.1em;
            font-weight: 500;
            color: #333;
        }
        .color-palette {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
        .color-swatch {
            width: 50px;
            height: 50px;
            border-radius: 5px;
            border: 1px solid #ddd;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.7em;
            color: white;
            text-shadow: 0 0 2px rgba(0,0,0,0.5);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📸 TAKAC PHOTO Analyzer</h1>

        <div class="upload-area">
            <div id="drop-zone">
                <p>🎯 Drag & drop your image here</p>
                <p style="font-size: 0.9em; color: #999;">or click to select</p>
                <input type="file" id="file-input" accept="image/*,.heic,.heif">
            </div>

            <div class="preview" id="preview">
                <h3>Preview:</h3>
                <img id="preview-img" src="" alt="Preview">
            </div>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p style="margin-top: 15px;">Analyzing image...</p>
            </div>
        </div>

        <div class="results" id="results">
            <h2>Analysis Results</h2>

            <div class="result-section" id="basic-info">
                <h3>📊 Basic Information</h3>
                <div class="metadata-grid" id="basic-grid"></div>
            </div>

            <div class="result-section" id="camera-info">
                <h3>📷 Camera Settings</h3>
                <div class="metadata-grid" id="camera-grid"></div>
            </div>

            <div class="result-section" id="location-info">
                <h3>📍 Location</h3>
                <div class="metadata-grid" id="location-grid"></div>
            </div>

            <div class="result-section" id="vision-info">
                <h3>👁️ Vision Analysis</h3>
                <div class="metadata-grid" id="vision-grid"></div>
            </div>

            <div class="result-section" id="color-info">
                <h3>🎨 Color Palette</h3>
                <div class="color-palette" id="color-palette"></div>
            </div>

            <div class="result-section">
                <h3>🔍 Full JSON Result</h3>
                <div class="json-view">
                    <pre id="json-result"></pre>
                </div>
            </div>
        </div>
    </div>

    <script>
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');
        const preview = document.getElementById('preview');
        const previewImg = document.getElementById('preview-img');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');

        // Click to select file
        dropZone.addEventListener('click', () => fileInput.click());

        // Handle file selection
        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) handleFile(file);
        });

        // Drag and drop
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('drag-over');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('drag-over');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
            const file = e.dataTransfer.files[0];
            if (file) handleFile(file);
        });

        function handleFile(file) {
            // Show preview
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImg.src = e.target.result;
                preview.style.display = 'block';
            };
            reader.readAsDataURL(file);

            // Upload and analyze
            uploadFile(file);
        }

        async function uploadFile(file) {
            const formData = new FormData();
            formData.append('file', file);

            loading.style.display = 'block';
            results.style.display = 'none';

            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();
                loading.style.display = 'none';

                if (response.ok) {
                    displayResults(data);
                } else {
                    alert('Error: ' + (data.error || 'Analysis failed'));
                }
            } catch (error) {
                loading.style.display = 'none';
                alert('Error: ' + error.message);
            }
        }

        function displayResults(data) {
            results.style.display = 'block';

            // Basic info
            const basicGrid = document.getElementById('basic-grid');
            basicGrid.innerHTML = '';
            if (data.services['metadata-extractor']) {
                const basic = data.services['metadata-extractor'].result.basic_info;
                const props = basic.image_properties;

                addMetadataItem(basicGrid, 'Format', basic.format);
                addMetadataItem(basicGrid, 'Resolution', `${props.width}×${props.height}`);
                addMetadataItem(basicGrid, 'Orientation', props.orientation);
                addMetadataItem(basicGrid, 'File Size', formatBytes(basic.file_size_bytes));
            }

            // Camera info
            const cameraGrid = document.getElementById('camera-grid');
            cameraGrid.innerHTML = '';
            if (data.services['metadata-extractor']) {
                const exif = data.services['metadata-extractor'].result.exif;

                if (exif.camera_make) addMetadataItem(cameraGrid, 'Camera', `${exif.camera_make} ${exif.camera_model}`);
                if (exif.iso) addMetadataItem(cameraGrid, 'ISO', exif.iso);
                if (exif.f_number) addMetadataItem(cameraGrid, 'Aperture', `f/${exif.f_number}`);
                if (exif.exposure_time_seconds) addMetadataItem(cameraGrid, 'Shutter', `1/${Math.round(1/exif.exposure_time_seconds)}`);
            }

            // Location info
            const locationGrid = document.getElementById('location-grid');
            locationGrid.innerHTML = '';
            if (data.services['metadata-extractor']) {
                const gps = data.services['metadata-extractor'].result.gps;

                if (gps.latitude) addMetadataItem(locationGrid, 'Latitude', gps.latitude.toFixed(6));
                if (gps.longitude) addMetadataItem(locationGrid, 'Longitude', gps.longitude.toFixed(6));
                if (gps.altitude_m) addMetadataItem(locationGrid, 'Altitude', `${gps.altitude_m.toFixed(2)} m`);
            }

            // Vision info
            const visionGrid = document.getElementById('vision-grid');
            visionGrid.innerHTML = '';
            if (data.services['vision-enhanced']) {
                const vision = data.services['vision-enhanced'].result;

                addMetadataItem(visionGrid, 'Scene', vision.scene_type);
                addMetadataItem(visionGrid, 'People', vision.people_count);
                addMetadataItem(visionGrid, 'Objects', vision.objects.length);
                addMetadataItem(visionGrid, 'Provider', vision.provider);
            }

            // Color palette
            const colorPalette = document.getElementById('color-palette');
            colorPalette.innerHTML = '';
            if (data.services['vision-analyzer']) {
                const colors = data.services['vision-analyzer'].result.color_palette;
                colors.slice(0, 5).forEach(color => {
                    const swatch = document.createElement('div');
                    swatch.className = 'color-swatch';
                    swatch.style.backgroundColor = color.hex;
                    swatch.textContent = Math.round(color.percentage * 100) + '%';
                    colorPalette.appendChild(swatch);
                });
            }

            // Full JSON
            document.getElementById('json-result').textContent = JSON.stringify(data, null, 2);
        }

        function addMetadataItem(container, label, value) {
            const item = document.createElement('div');
            item.className = 'metadata-item';
            item.innerHTML = `
                <div class="metadata-label">${label}</div>
                <div class="metadata-value">${value}</div>
            `;
            container.appendChild(item);
        }

        function formatBytes(bytes) {
            if (bytes === 0) return '0 Bytes';
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
        }
    </script>
</body>
</html>''')

    print("🚀 Starting TAKAC PHOTO web server...")
    print("📍 Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)