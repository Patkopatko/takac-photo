#!/usr/bin/env python3
"""
Unit tests for Professional Image Analyzer
Tests all edge cases as per requirements
"""

import unittest
import json
import tempfile
import os
from PIL import Image
import numpy as np
from professional_analyzer import (
    ProfessionalImageAnalyzer,
    X11ColorMapper,
    JSONSchemaValidator,
    Orientation,
    MetadataReason
)

class TestColorMapping(unittest.TestCase):
    """Test X11 color mapping"""

    def setUp(self):
        self.mapper = X11ColorMapper()

    def test_exact_colors(self):
        """Test exact color matches"""
        self.assertEqual(self.mapper.get_color_name(255, 255, 255), "white")
        self.assertEqual(self.mapper.get_color_name(0, 0, 0), "black")
        self.assertEqual(self.mapper.get_color_name(255, 0, 0), "red")
        self.assertEqual(self.mapper.get_color_name(0, 255, 0), "lime")
        self.assertEqual(self.mapper.get_color_name(192, 192, 192), "silver")

    def test_near_colors(self):
        """Test near color matches"""
        # Near white
        self.assertEqual(self.mapper.get_color_name(250, 250, 250), "snow")
        # Near gray
        self.assertIn(self.mapper.get_color_name(130, 130, 130), ["gray", "darkgray"])

    def test_unknown_colors(self):
        """Test colors too far from known palette"""
        # Very specific color far from all
        result = self.mapper.get_color_name(123, 87, 201)
        self.assertIn(result, ["unknown", "purple", "violet"])  # Could match purple-ish

    def test_gray_not_red(self):
        """Specific test: gray should never be 'red'"""
        grays = [
            (128, 128, 128),
            (192, 192, 192),
            (211, 211, 211),
            (169, 169, 169),
            (200, 200, 200)
        ]
        for r, g, b in grays:
            color = self.mapper.get_color_name(r, g, b)
            self.assertNotEqual(color, "red", f"Gray {r},{g},{b} was mapped to red!")
            self.assertIn("gray", color.lower())

class TestMetadataExtraction(unittest.TestCase):
    """Test metadata extraction with various scenarios"""

    def setUp(self):
        self.analyzer = ProfessionalImageAnalyzer()

    def create_test_image(self, width=100, height=100, format="JPEG", with_exif=False):
        """Create a test image file"""
        img = Image.new('RGB', (width, height), color='white')

        if with_exif:
            # Add basic EXIF
            from PIL import ExifTags
            exif = img.getexif()
            exif[ExifTags.Base.Make] = "TestCamera"
            exif[ExifTags.Base.Model] = "TestModel"

        temp_file = tempfile.NamedTemporaryFile(suffix=f".{format.lower()}", delete=False)
        img.save(temp_file.name, format=format)
        return temp_file.name

    def test_missing_exif(self):
        """Test image without EXIF data"""
        img_path = self.create_test_image(with_exif=False)

        try:
            metadata = self.analyzer.extract_metadata(img_path)

            # Check structure
            self.assertIn("basic_info", metadata)
            self.assertIn("exif", metadata)
            self.assertIn("gps", metadata)
            self.assertIn("metadata_preserved", metadata)
            self.assertIn("metadata_reason", metadata)

            # Check EXIF is None/null
            self.assertIsNone(metadata["exif"]["camera_make"])
            self.assertIsNone(metadata["exif"]["camera_model"])
            self.assertIsNone(metadata["exif"]["iso"])

            # Check metadata not preserved
            self.assertEqual(metadata["metadata_preserved"], False)
            self.assertIn(metadata["metadata_reason"],
                         [MetadataReason.STRIPPED_EXPORT.value,
                          MetadataReason.UNKNOWN.value])

        finally:
            os.unlink(img_path)

    def test_orientation_detection(self):
        """Test different image orientations"""
        test_cases = [
            (100, 200, Orientation.PORTRAIT.value),
            (200, 100, Orientation.LANDSCAPE.value),
            (100, 100, Orientation.SQUARE.value)
        ]

        for width, height, expected_orientation in test_cases:
            img_path = self.create_test_image(width, height)

            try:
                metadata = self.analyzer.extract_metadata(img_path)
                actual = metadata["basic_info"]["image_properties"]["orientation"]
                self.assertEqual(actual, expected_orientation,
                               f"{width}x{height} should be {expected_orientation}")
            finally:
                os.unlink(img_path)

    def test_dimension_consistency(self):
        """Test that dimensions are consistent throughout"""
        img_path = self.create_test_image(width=1920, height=1080)

        try:
            metadata = self.analyzer.extract_metadata(img_path)

            # Check dimensions exist and are consistent
            props = metadata["basic_info"]["image_properties"]
            self.assertEqual(props["width"], 1920)
            self.assertEqual(props["height"], 1080)

            # Ensure no conflicting dimensions elsewhere
            self.assertNotIn("exif_image_width", metadata["exif"])
            self.assertNotIn("exif_image_height", metadata["exif"])

        finally:
            os.unlink(img_path)

    def test_null_vs_zero_handling(self):
        """Test proper null vs 0 handling"""
        img_path = self.create_test_image(with_exif=False)

        try:
            metadata = self.analyzer.extract_metadata(img_path)

            # EXIF values should be None when missing, not 0
            self.assertIsNone(metadata["exif"]["iso"])
            self.assertIsNone(metadata["exif"]["f_number"])
            self.assertIsNone(metadata["exif"]["focal_length_mm"])

            # GPS should be None when missing, not 0.0
            self.assertIsNone(metadata["gps"]["latitude"])
            self.assertIsNone(metadata["gps"]["longitude"])
            self.assertIsNone(metadata["gps"]["altitude_m"])

            # File size should be actual value > 0
            self.assertGreater(metadata["basic_info"]["file_size_bytes"], 0)

        finally:
            os.unlink(img_path)

class TestJSONSchemaValidation(unittest.TestCase):
    """Test JSON Schema validation"""

    def setUp(self):
        self.validator = JSONSchemaValidator()
        self.analyzer = ProfessionalImageAnalyzer()

    def test_valid_output(self):
        """Test that valid output passes validation"""
        valid_data = {
            "schema_version": "2.2.0",
            "source": "test",
            "image_id": "test-id",
            "checksum": "test-checksum",
            "request_id": "R-test",
            "status": "done",
            "processing_timestamp": "2025-09-29T20:00:00",
            "services": {
                "test-service": {
                    "status": "success",
                    "result": {"test": "data"},
                    "metrics": {
                        "processing_time_ms": 100,
                        "service_version": "1.0.0"
                    },
                    "error": None
                }
            },
            "derivative_insights": {
                "version": 1,
                "groups": []
            },
            "progress": {
                "done": 1,
                "expected": 1,
                "success_rate": 1.0
            }
        }

        self.assertTrue(self.validator.validate(valid_data))

    def test_missing_required_fields(self):
        """Test that missing required fields fail validation"""
        invalid_data = {
            "schema_version": "2.2.0",
            "source": "test"
            # Missing other required fields
        }

        self.assertFalse(self.validator.validate(invalid_data))

    def test_invalid_enum_values(self):
        """Test that invalid enum values fail"""
        data = {
            "schema_version": "2.2.0",
            "source": "test",
            "image_id": "test-id",
            "checksum": "test-checksum",
            "request_id": "R-test",
            "status": "invalid_status",  # Invalid enum
            "processing_timestamp": "2025-09-29T20:00:00",
            "services": {},
            "derivative_insights": {"version": 1, "groups": []},
            "progress": {"done": 1, "expected": 1, "success_rate": 1.0}
        }

        self.assertFalse(self.validator.validate(data))

class TestDeduplication(unittest.TestCase):
    """Test that there are no duplicate sections"""

    def setUp(self):
        self.analyzer = ProfessionalImageAnalyzer()

    def test_no_duplicate_raw_analysis(self):
        """Ensure no duplicate raw_analysis sections"""
        img_path = TestMetadataExtraction().create_test_image()

        try:
            result = self.analyzer.analyze_image(img_path)

            # Convert to JSON string
            json_str = json.dumps(result)

            # Count occurrences of "raw_analysis"
            count = json_str.count('"raw_analysis"')

            self.assertEqual(count, 0, f"Found {count} occurrences of raw_analysis, expected 0")

        finally:
            os.unlink(img_path)

    def test_single_dimension_source(self):
        """Ensure dimensions only come from one source"""
        img_path = TestMetadataExtraction().create_test_image(width=800, height=600)

        try:
            result = self.analyzer.analyze_image(img_path)

            # Find all width/height references
            json_str = json.dumps(result)

            # Should only have dimensions in basic_info.image_properties
            self.assertIn('"width": 800', json_str)
            self.assertIn('"height": 600', json_str)

            # Count total occurrences
            width_count = json_str.count('"width": 800')
            height_count = json_str.count('"height": 600')

            # Should appear exactly once each
            self.assertEqual(width_count, 1, "Width should appear exactly once")
            self.assertEqual(height_count, 1, "Height should appear exactly once")

        finally:
            os.unlink(img_path)

class TestVisionMetrics(unittest.TestCase):
    """Test vision metrics calculation"""

    def setUp(self):
        self.analyzer = ProfessionalImageAnalyzer()

    def test_metrics_vs_semantics_separation(self):
        """Test that metrics and interpretations are separated"""
        img_path = TestMetadataExtraction().create_test_image()

        try:
            vision = self.analyzer.analyze_vision_metrics(img_path)

            # Check structure
            self.assertIn("quantitative_metrics", vision)
            self.assertIn("buckets", vision)
            self.assertIn("color_palette", vision)

            # Metrics should be numbers only
            metrics = vision["quantitative_metrics"]
            for key, value in metrics.items():
                self.assertIsInstance(value, (int, float),
                                    f"{key} should be numeric, got {type(value)}")

            # Buckets should be strings only
            buckets = vision["buckets"]
            for key, value in buckets.items():
                self.assertIsInstance(value, str,
                                    f"{key} should be string, got {type(value)}")

            # No mixing - metrics shouldn't have strings like "bright"
            for value in metrics.values():
                self.assertNotIsInstance(value, str)

        finally:
            os.unlink(img_path)

if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)