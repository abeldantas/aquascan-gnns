#!/bin/bash
# Create a clean package for Colab upload

echo "📦 Packaging Aquascan for Colab..."

# Create temp directory
TEMP_DIR="/tmp/aquascan-colab"
rm -rf $TEMP_DIR
mkdir -p $TEMP_DIR

# Copy only essential files (skip data and venv)
cp -r /Users/adantas/dev/phd/aquascan-gnns/aquascan $TEMP_DIR/
cp -r /Users/adantas/dev/phd/aquascan-gnns/configs $TEMP_DIR/
cp -r /Users/adantas/dev/phd/aquascan-gnns/scripts $TEMP_DIR/
cp -r /Users/adantas/dev/phd/aquascan-gnns/docs $TEMP_DIR/
cp /Users/adantas/dev/phd/aquascan-gnns/requirements.txt $TEMP_DIR/
cp /Users/adantas/dev/phd/aquascan-gnns/README.md $TEMP_DIR/
cp /Users/adantas/dev/phd/aquascan-gnns/QUICK_COMMANDS.md $TEMP_DIR/

# Create directories for data
mkdir -p $TEMP_DIR/data/raw
mkdir -p $TEMP_DIR/data/processed
mkdir -p $TEMP_DIR/results
mkdir -p $TEMP_DIR/checkpoints

# Create the archive
cd /tmp
tar -czf aquascan-colab.tar.gz aquascan-colab/

# Move to Downloads
mv aquascan-colab.tar.gz ~/Downloads/

echo "✅ Package ready at: ~/Downloads/aquascan-colab.tar.gz"
echo "📏 Size: $(du -h ~/Downloads/aquascan-colab.tar.gz | cut -f1)"
echo ""
echo "🚀 Next: Upload this file to your Google Drive or Colab directly"
