#!/usr/bin/env python3
"""
Image Analysis Tool - Analyzuje obrázky a vracia štruktúrovaný JSON výstup
Používa: OpenAI GPT-4V alebo Claude Vision API
Autor: Claude
"""

import json
import base64
import requests
import os
import sys
from datetime import datetime
from PIL import Image, ExifTags
from pathlib import Path

# Registruje HEIC podporu
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    print("Warning: pillow_heif nie je nainštalované, HEIC súbory nebudú podporované")

class ImageAnalyzer:
    def __init__(self, api_key=None, provider="openai"):
        """
        Initialize analyzer
        provider: "openai" alebo "anthropic"
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY')
        self.provider = provider.lower()

        if not self.api_key:
            raise ValueError("API kľúč nie je nastavený! Nastav OPENAI_API_KEY alebo ANTHROPIC_API_KEY")

    def encode_image(self, image_path):
        """Konvertuje obrázok na base64"""
        try:
            # Ak je HEIC, konvertuj na JPEG
            if str(image_path).lower().endswith('.heic'):
                with Image.open(image_path) as img:
                    # Konvertuj na RGB ak nie je
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    # Uložiť do pamäte ako JPEG
                    import io
                    output = io.BytesIO()
                    img.save(output, format='JPEG', quality=95)
                    output.seek(0)
                    return base64.b64encode(output.read()).decode('utf-8')
            else:
                # Normálny súbor
                with open(image_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            raise Exception(f"Chyba pri načítaní obrázka: {e}")

    def get_image_metadata(self, image_path):
        """Získa základné metadata obrázka"""
        try:
            with Image.open(image_path) as img:
                width, height = img.size

                # Skús získať EXIF dáta
                exif_data = {}
                if hasattr(img, '_getexif') and img._getexif() is not None:
                    exif = img._getexif()
                    for tag_id, value in exif.items():
                        tag = ExifTags.TAGS.get(tag_id, tag_id)
                        exif_data[tag] = value

                return {
                    "width": width,
                    "height": height,
                    "format": img.format,
                    "mode": img.mode,
                    "exif": exif_data
                }
        except Exception as e:
            return {"error": f"Chyba pri čítaní metadát: {e}"}

    def create_analysis_prompt(self):
        """Vytvorí detailný prompt pre analýzu obrázka"""
        return """
Analyzuj tento obrázok veľmi detailne a vráť výsledky v presnom JSON formáte.

VRÁŤ PRESNE TENTO JSON FORMÁT - nič viac, nič menej:

{
  "Request ID": "R-generated-id",
  "Status": "DONE",
  "Message": "Image accepted for processing",
  "Progress": "3/3 services finished",
  "Scene insights": "The user captured... \\n The user captured...",
  "Object insights": "The user captured... \\n The user captured... \\n The user captured...",
  "People insights": "The user captured X people. \\n The user captured...",
  "Environment insights": "The user captured... \\n The user captured...",
  "Capture insights": "The user captured the photo on YYYY-MM-DD at HH:MM:SS.",
  "Quality insights": "The user captured a resolution image with dimensions X by Y pixels. \\n The user captured...",
  "Composition insights": "The user captured... \\n The user captured...",
  "Safety insights": "The user captured content that appears safe for work."
}

DÔLEŽITÉ:
- Každý insight je STRING, nie array
- Viacero insights oddeľ \\n
- Každý insight začína "The user captured..."
- Buď veľmi špecifický a detailný
- Zachovaj presne tento formát vrátane kľúčov s veľkými písmenami
"""

    def analyze_with_openai(self, image_base64):
        """Analyzuje obrázok cez OpenAI GPT-4V"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.create_analysis_prompt()
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.1
        }

        response = requests.post("https://api.openai.com/v1/chat/completions",
                               headers=headers, json=payload)

        if response.status_code != 200:
            raise Exception(f"OpenAI API chyba: {response.status_code} - {response.text}")

        return response.json()['choices'][0]['message']['content']

    def analyze_with_anthropic(self, image_base64):
        """Analyzuje obrázok cez Claude Vision"""
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }

        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 2000,
            "temperature": 0.1,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.create_analysis_prompt()
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64
                            }
                        }
                    ]
                }
            ]
        }

        response = requests.post("https://api.anthropic.com/v1/messages",
                               headers=headers, json=payload)

        if response.status_code != 200:
            raise Exception(f"Anthropic API chyba: {response.status_code} - {response.text}")

        return response.json()['content'][0]['text']

    def parse_json_from_response(self, response_text):
        """Extraktuje a parsuje JSON z odpovede"""
        try:
            # Skús nájsť JSON v odpovedi
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx == -1 or end_idx == 0:
                raise ValueError("JSON nenájdený v odpovedi")

            json_str = response_text[start_idx:end_idx]
            return json.loads(json_str)

        except json.JSONDecodeError as e:
            # Ak sa nepodarí parsovať, vráť raw text
            return {
                "error": "Chyba pri parsovaní JSON",
                "raw_response": response_text,
                "json_error": str(e)
            }

    def analyze_image(self, image_path):
        """Hlavná funkcia - analyzuje obrázok"""
        print(f"Analyzujem obrázok: {image_path}")

        # Skontroluj či súbor existuje
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Obrázok nenájdený: {image_path}")

        # Získaj metadata
        metadata = self.get_image_metadata(image_path)
        print(f"Rozlíšenie: {metadata.get('width')}x{metadata.get('height')}")

        # Konvertuj na base64
        image_base64 = self.encode_image(image_path)
        print(f"Obrázok konvertovaný na base64 ({len(image_base64)} znakov)")

        # Analyzuj podľa providera
        print(f"Analyzujem cez {self.provider.upper()}...")

        if self.provider == "openai":
            response = self.analyze_with_openai(image_base64)
        elif self.provider == "anthropic":
            response = self.analyze_with_anthropic(image_base64)
        else:
            raise ValueError(f"Nepodporovaný provider: {self.provider}")

        # Parsuj JSON
        result = self.parse_json_from_response(response)

        # Získaj skutočné EXIF metadata z mdls
        try:
            import subprocess
            mdls_result = subprocess.run(['mdls', str(image_path)],
                                       capture_output=True, text=True)
            exif_info = mdls_result.stdout

            # Extraktuj kľúčové informácie
            date_created = self.extract_mdls_value(exif_info, 'kMDItemContentCreationDate')
            device_make = self.extract_mdls_value(exif_info, 'kMDItemAcquisitionMake')
            device_model = self.extract_mdls_value(exif_info, 'kMDItemAcquisitionModel')
            latitude = self.extract_mdls_value(exif_info, 'kMDItemLatitude')
            longitude = self.extract_mdls_value(exif_info, 'kMDItemLongitude')
            altitude = self.extract_mdls_value(exif_info, 'kMDItemAltitude')
            iso = self.extract_mdls_value(exif_info, 'kMDItemISOSpeed')
            focal_length = self.extract_mdls_value(exif_info, 'kMDItemFocalLength35mm')
            aperture = self.extract_mdls_value(exif_info, 'kMDItemFNumber')

            # Aktualizuj capture insights s reálnymi dátami
            if 'Capture insights' in result:
                capture_info = f"The user captured the photo on {date_created}"
                if device_make and device_model:
                    capture_info += f" using {device_make} {device_model}"
                if latitude and longitude:
                    capture_info += f" at GPS coordinates {latitude}, {longitude}"
                if altitude:
                    capture_info += f" at {altitude} meters above sea level"
                if iso:
                    capture_info += f" with ISO {iso}"
                if focal_length:
                    capture_info += f" and {focal_length}mm focal length"
                if aperture:
                    capture_info += f" at f/{aperture}"

                result['Capture insights'] = capture_info

        except Exception as e:
            print(f"Warning: Nepodarilo sa získať EXIF data: {e}")

        return result

    def extract_mdls_value(self, mdls_output, key):
        """Extraktuje hodnotu z mdls výstupu"""
        try:
            lines = mdls_output.split('\n')
            for line in lines:
                if key in line:
                    parts = line.split('=', 1)
                    if len(parts) == 2:
                        value = parts[1].strip().strip('"')
                        # Vyčisti hodnotu
                        if value.startswith('(') and value.endswith(')'):
                            return None
                        if value == '(null)':
                            return None
                        return value
            return None
        except:
            return None

def main():
    """CLI rozhranie"""
    if len(sys.argv) < 2:
        print("Použitie: python image_analyzer.py <cesta_k_obrazku> [provider]")
        print("Provider: 'openai' (default) alebo 'anthropic'")
        sys.exit(1)

    image_path = sys.argv[1]
    provider = sys.argv[2] if len(sys.argv) > 2 else "openai"

    try:
        analyzer = ImageAnalyzer(provider=provider)
        result = analyzer.analyze_image(image_path)

        # Vypíš JSON výsledok
        print("\n" + "="*50)
        print("JSON VÝSLEDOK:")
        print("="*50)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    except Exception as e:
        print(f"Chyba: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()