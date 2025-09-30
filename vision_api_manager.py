#!/usr/bin/env python3
"""
Vision API Manager with multiple provider support
Supports: OpenAI GPT-4V, Anthropic Claude, Google Gemini, Local models
"""

import json
import base64
import requests
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VisionResult:
    """Standardized vision analysis result"""
    success: bool
    provider: str
    scene_type: str = "unknown"
    people_count: int = 0
    face_count: int = 0
    objects: List[Dict] = None
    text_detected: List[str] = None
    description: str = ""
    confidence: float = 0.0
    raw_response: Dict = None
    error: str = None

    def __post_init__(self):
        if self.objects is None:
            self.objects = []
        if self.text_detected is None:
            self.text_detected = []

class VisionProvider(ABC):
    """Abstract base class for vision providers"""

    @abstractmethod
    def analyze_image(self, image_path: str) -> VisionResult:
        pass

    def encode_image_base64(self, image_path: str) -> str:
        """Encode image to base64"""
        from PIL import Image

        try:
            # Handle HEIC
            if image_path.lower().endswith('.heic'):
                try:
                    from pillow_heif import register_heif_opener
                    register_heif_opener()
                except ImportError:
                    logger.warning("HEIC support not available")

                with Image.open(image_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    import io
                    buffer = io.BytesIO()
                    img.save(buffer, format='JPEG', quality=95)
                    buffer.seek(0)
                    return base64.b64encode(buffer.read()).decode('utf-8')
            else:
                with open(image_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            return ""

class OpenAIVisionProvider(VisionProvider):
    """OpenAI GPT-4 Vision provider"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.base_url = "https://api.openai.com/v1/chat/completions"

    def analyze_image(self, image_path: str) -> VisionResult:
        if not self.api_key:
            return VisionResult(success=False, provider="openai", error="No API key")

        try:
            base64_image = self.encode_image_base64(image_path)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Analyze this image and provide:
                        1. Scene type (beach, indoor, street, nature, etc)
                        2. Number of people and faces visible
                        3. List of objects with descriptions
                        4. Any text visible in the image
                        5. Brief description

                        Format as JSON with keys: scene_type, people_count, face_count, objects, text, description"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }]

            payload = {
                "model": "gpt-4-vision-preview",
                "messages": messages,
                "max_tokens": 1000,
                "temperature": 0.1
            }

            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)

            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']

                # Parse JSON from response
                try:
                    # Extract JSON
                    start = content.find('{')
                    end = content.rfind('}') + 1
                    if start >= 0 and end > 0:
                        data = json.loads(content[start:end])

                        return VisionResult(
                            success=True,
                            provider="openai",
                            scene_type=data.get('scene_type', 'unknown'),
                            people_count=data.get('people_count', 0),
                            face_count=data.get('face_count', 0),
                            objects=data.get('objects', []),
                            text_detected=data.get('text', []),
                            description=data.get('description', ''),
                            confidence=0.9,
                            raw_response=data
                        )
                except json.JSONDecodeError:
                    pass

                # Fallback to text parsing
                return VisionResult(
                    success=True,
                    provider="openai",
                    description=content,
                    confidence=0.5
                )
            else:
                error = f"API error: {response.status_code}"
                logger.error(error)
                return VisionResult(success=False, provider="openai", error=error)

        except Exception as e:
            logger.error(f"OpenAI Vision error: {e}")
            return VisionResult(success=False, provider="openai", error=str(e))

class AnthropicVisionProvider(VisionProvider):
    """Anthropic Claude Vision provider"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.base_url = "https://api.anthropic.com/v1/messages"

    def analyze_image(self, image_path: str) -> VisionResult:
        if not self.api_key:
            return VisionResult(success=False, provider="anthropic", error="No API key")

        try:
            base64_image = self.encode_image_base64(image_path)

            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"
            }

            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this image: What do you see? Count people, identify objects, read any text."
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_image
                        }
                    }
                ]
            }]

            payload = {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 1000,
                "temperature": 0.1,
                "messages": messages
            }

            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)

            if response.status_code == 200:
                result = response.json()
                content = result['content'][0]['text']

                return VisionResult(
                    success=True,
                    provider="anthropic",
                    description=content,
                    confidence=0.9
                )
            else:
                error = f"API error: {response.status_code}"
                logger.error(error)
                return VisionResult(success=False, provider="anthropic", error=error)

        except Exception as e:
            logger.error(f"Anthropic Vision error: {e}")
            return VisionResult(success=False, provider="anthropic", error=str(e))

class LocalVisionProvider(VisionProvider):
    """Local model provider using YOLO or similar"""

    def __init__(self):
        self.model_loaded = False
        self.model = None
        self._load_model()

    def _load_model(self):
        """Try to load a local model"""
        try:
            # Try YOLO
            try:
                from ultralytics import YOLO
                self.model = YOLO('yolov8n.pt')
                self.model_loaded = True
                logger.info("YOLO model loaded")
            except ImportError:
                pass

            # Try other models...
            if not self.model_loaded:
                logger.warning("No local model available")

        except Exception as e:
            logger.error(f"Error loading local model: {e}")

    def analyze_image(self, image_path: str) -> VisionResult:
        if not self.model_loaded:
            return VisionResult(success=False, provider="local", error="No local model available")

        try:
            # Run YOLO detection
            results = self.model(image_path)

            objects = []
            people_count = 0

            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        name = self.model.names[cls]

                        objects.append({
                            "name": name,
                            "confidence": conf
                        })

                        if name == "person":
                            people_count += 1

            return VisionResult(
                success=True,
                provider="local",
                people_count=people_count,
                objects=objects,
                confidence=0.7
            )

        except Exception as e:
            logger.error(f"Local model error: {e}")
            return VisionResult(success=False, provider="local", error=str(e))

class VisionAPIManager:
    """Manager for multiple vision API providers with fallback"""

    def __init__(self):
        self.providers = []
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize available providers in priority order"""

        # Try OpenAI first
        if os.getenv('OPENAI_API_KEY'):
            self.providers.append(OpenAIVisionProvider())
            logger.info("OpenAI Vision provider initialized")

        # Try Anthropic
        if os.getenv('ANTHROPIC_API_KEY'):
            self.providers.append(AnthropicVisionProvider())
            logger.info("Anthropic Vision provider initialized")

        # Always add local as fallback
        self.providers.append(LocalVisionProvider())

        if not self.providers:
            logger.warning("No vision providers available")

    def analyze_image(self, image_path: str) -> VisionResult:
        """
        Analyze image using available providers with fallback
        """
        if not os.path.exists(image_path):
            return VisionResult(success=False, provider="none", error="File not found")

        # Try each provider in order
        for provider in self.providers:
            logger.info(f"Trying {provider.__class__.__name__}")
            result = provider.analyze_image(image_path)

            if result.success:
                logger.info(f"Success with {result.provider}")
                return result

        # All providers failed
        return VisionResult(
            success=False,
            provider="none",
            error="All vision providers failed"
        )

    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return [p.__class__.__name__ for p in self.providers]

# Example usage
def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python vision_api_manager.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    # Initialize manager
    manager = VisionAPIManager()
    print(f"Available providers: {manager.get_available_providers()}")

    # Analyze image
    print(f"\nAnalyzing: {image_path}")
    result = manager.analyze_image(image_path)

    if result.success:
        print(f"\n✅ Success with {result.provider}")
        print(f"Scene: {result.scene_type}")
        print(f"People: {result.people_count}")
        print(f"Objects: {len(result.objects)}")
        if result.text_detected:
            print(f"Text: {result.text_detected}")
        print(f"Description: {result.description[:200]}")
    else:
        print(f"\n❌ Failed: {result.error}")

if __name__ == "__main__":
    main()