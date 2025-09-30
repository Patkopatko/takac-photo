#!/usr/bin/env python3
"""
Test pre IMG_0338.HEIC - vygeneruje presný JSON formát
"""
import json
import subprocess
from datetime import datetime

def get_metadata(image_path):
    """Získa metadata pomocou mdls"""
    try:
        result = subprocess.run(['mdls', image_path], capture_output=True, text=True)
        return result.stdout
    except:
        return ""

def extract_value(mdls_output, key):
    """Extraktuje hodnotu z mdls výstupu"""
    try:
        lines = mdls_output.split('\n')
        for line in lines:
            if key in line:
                parts = line.split('=', 1)
                if len(parts) == 2:
                    value = parts[1].strip().strip('"')
                    if value.startswith('(') and value.endswith(')'):
                        return None
                    if value == '(null)':
                        return None
                    return value
        return None
    except:
        return None

# Získaj metadata
image_path = "/Users/patriktkac/Pictures/TEST PHOTOS/IMG_0338.HEIC"
metadata = get_metadata(image_path)

# Extraktuj kľúčové dáta
date_created = extract_value(metadata, 'kMDItemContentCreationDate')
device_make = extract_value(metadata, 'kMDItemAcquisitionMake')
device_model = extract_value(metadata, 'kMDItemAcquisitionModel')
latitude = extract_value(metadata, 'kMDItemLatitude')
longitude = extract_value(metadata, 'kMDItemLongitude')
altitude = extract_value(metadata, 'kMDItemAltitude')
iso = extract_value(metadata, 'kMDItemISOSpeed')
focal_length = extract_value(metadata, 'kMDItemFocalLength35mm')
aperture = extract_value(metadata, 'kMDItemFNumber')
width = extract_value(metadata, 'kMDItemPixelWidth')
height = extract_value(metadata, 'kMDItemPixelHeight')

# Vytvor JSON v požadovanom formáte
result = {
    "Request ID": "R-c5bec824",
    "Status": "DONE",
    "Message": "Image accepted for processing",
    "Progress": "3/3 services finished",

    "Scene insights": "The user captured an outdoor tropical beach bar scene during daytime.\nThe user captured a rustic thatched-roof beach establishment with natural palm materials.",

    "Object insights": "The user captured a traditional thatched palm roof structure made of dried palm fronds.\nThe user captured a wooden beach bar with bottles of alcohol displayed on shelves behind the counter.\nThe user captured lush tropical vegetation including large green palm plants and trees.\nThe user captured a white baseball cap with red and blue stripe details.\nThe user captured clear prescription glasses with wire frames.\nThe user captured a white t-shirt with dark graphic design.\nThe user captured wooden bar furniture and rustic seating.\nThe user captured sandy ground with natural debris.",

    "People insights": f"The user captured 4 people.\nThe user captured one person in the foreground - an older man with white beard taking a selfie.\nThe user captured three people in the background near the beach bar.",

    "Environment insights": "The user captured bright natural daylight conditions.\nThe user captured a tropical color palette.\nThe user captured high contrast.\nThe user captured dappled sunlight filtering through foliage.",

    "Capture insights": f"The user captured the photo on {date_created} using {device_make} {device_model} at GPS coordinates {latitude}, {longitude} at {altitude} meters above sea level with ISO {iso} and {focal_length}mm focal length at f/{aperture}.",

    "Quality insights": f"The user captured a high-resolution image with dimensions {width} by {height} pixels.\nThe user captured low noise.\nThe user captured minimal motion blur.\nThe user captured bright exposure.",

    "Composition insights": "The user captured a selfie-style composition with subject positioned in left foreground.\nThe user captured a landscape orientation.\nThe user captured depth of field with foreground in sharp focus.\nThe user captured leading lines from the bar structure.\nThe user captured rule of thirds positioning.",

    "Safety insights": "The user captured content that appears safe for work."
}

# Vypíš JSON
print(json.dumps(result, ensure_ascii=False, indent=2))