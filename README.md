# 📸 TAKAC PHOTO - Professional Image Analyzer

Advanced image analysis tool that extracts comprehensive metadata, performs visual analysis, and generates structured JSON reports from photos.

## 🌟 Features

- **Complete Metadata Extraction**
  - EXIF data (camera, settings, datetime)
  - GPS coordinates with location mapping
  - Full support for HEIC, JPEG, PNG formats
  - iPhone metadata preservation

- **Visual Analysis**
  - Color palette extraction with X11/web-safe naming
  - Brightness, contrast, saturation metrics
  - Sharpness and edge detection
  - Scene classification

- **Content Detection** (with API key)
  - People counting and face detection
  - Object recognition
  - Text extraction from images
  - Environment and lighting analysis

- **Professional JSON Output**
  - JSON Schema v2.2.0 validated
  - Consistent structure and typing
  - SHA256 checksums
  - Processing metrics

## 🚀 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/takac-photo.git
cd takac-photo

# Install dependencies
pip3 install -r requirements.txt

# Optional: Set up API key for content analysis
export OPENAI_API_KEY="your-key-here"
# or
export ANTHROPIC_API_KEY="your-key-here"
```

## 📦 Requirements

- Python 3.8+
- PIL (Pillow)
- NumPy
- SciPy
- pillow-heif (for HEIC support)
- jsonschema
- requests
- exifread

## 🔧 Usage

### Basic Analysis
```bash
python3 professional_analyzer.py /path/to/image.jpg
```

### With Vision API
```bash
# Set API key first
export OPENAI_API_KEY="your-key"

# Run analyzer
python3 professional_analyzer.py /path/to/image.heic > output.json
```

### Running Tests
```bash
python3 test_professional_analyzer.py
```

## 📊 Output Example

```json
{
  "schema_version": "2.2.0",
  "source": "professional-analyzer",
  "image_id": "unique-uuid",
  "checksum": "sha256-hash",
  "services": {
    "metadata-extractor": {
      "result": {
        "basic_info": {
          "format": "HEIC",
          "image_properties": {
            "width": 4032,
            "height": 3024,
            "orientation": "landscape"
          }
        },
        "exif": {
          "camera_make": "Apple",
          "camera_model": "iPhone 16 Pro Max",
          "iso": 32,
          "f_number": 1.9
        },
        "gps": {
          "latitude": -2.79578,
          "longitude": -40.51739,
          "altitude_m": 10.62
        }
      }
    },
    "vision-analyzer": {
      "result": {
        "quantitative_metrics": {
          "brightness_mean": 154.3,
          "sharpness_score": 0.244
        },
        "color_palette": [
          {
            "hex": "#dbe5fe",
            "color_name": "paleblue"
          }
        ]
      }
    }
  }
}
```

## 🏗️ Project Structure

```
takac-photo/
├── professional_analyzer.py          # Main analyzer with metadata extraction
├── professional_analyzer_with_vision.py  # Vision API integration
├── test_professional_analyzer.py     # Unit tests
├── requirements.txt                  # Python dependencies
├── setup.sh                          # Installation script
├── .gitignore                        # Git ignore rules
└── README.md                         # This file
```

## 🧪 Tested Scenarios

- ✅ HEIC files from iPhone with GPS
- ✅ Images without EXIF data
- ✅ Different orientations (portrait/landscape/square)
- ✅ Drag & drop stripped metadata
- ✅ Various color palettes
- ✅ Large resolution images (12MP+)

## 🔑 API Integration

### OpenAI GPT-4 Vision
```python
analyzer = ProfessionalImageAnalyzer(api_key="sk-...", provider="openai")
```

### Anthropic Claude Vision
```python
analyzer = ProfessionalImageAnalyzer(api_key="sk-ant-...", provider="anthropic")
```

## 📈 Performance

- Metadata extraction: ~40ms
- Vision analysis: ~3-4s
- Color palette: ~100ms
- Total processing: <5s for 12MP image

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Built with Claude Code (Anthropic)
- Uses OpenAI GPT-4 Vision for content analysis
- X11 color naming system
- EXIF standards implementation

## 📞 Contact

Patrik Tkáč - [@patriktkac](https://github.com/patriktkac)

Project Link: [https://github.com/patriktkac/takac-photo](https://github.com/patriktkac/takac-photo)

---
*Generated with ❤️ by Claude Code*