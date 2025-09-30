#!/bin/bash
# Inštalačný skript pre TAKAC PHOTO Image Analyzer

echo "🚀 Inštalujem TAKAC PHOTO Image Analyzer..."

# Skontroluj Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 nie je nainštalovaný!"
    exit 1
fi

echo "✅ Python3 nájdený"

# Inštaluj závislosti
echo "📦 Inštalujem závislosti..."
pip3 install -r requirements.txt

echo "✅ Závislosti nainštalované"

# Nastav executable práva
chmod +x image_analyzer.py

echo ""
echo "🎉 TAKAC PHOTO Image Analyzer je pripravený!"
echo ""
echo "📋 NÁVOD NA POUŽITIE:"
echo "1. Nastav API kľúč:"
echo "   export OPENAI_API_KEY='tvoj_kluc'"
echo "   alebo"
echo "   export ANTHROPIC_API_KEY='tvoj_kluc'"
echo ""
echo "2. Spusti analýzu:"
echo "   python3 image_analyzer.py /cesta/k/obrazku.jpg"
echo "   alebo"
echo "   python3 image_analyzer.py /cesta/k/obrazku.jpg anthropic"
echo ""
echo "✨ Hotovo!"