#!/bin/bash
# Quick setup to get Colab pipeline ready

echo "🚀 Preparing Aquascan for Colab..."

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: Run this from the aquascan-gnns root directory"
    exit 1
fi

# Kill any running processes
echo "🔪 Stopping any running processes..."
pkill -f "build_graphs" 2>/dev/null || true
pkill -f "aquascan" 2>/dev/null || true

# Add notebooks to git
echo "📓 Adding notebooks to git..."
git add notebooks/
git commit -m "Add Colab pipeline notebooks" 2>/dev/null || echo "Already committed"

echo ""
echo "✅ Ready for Colab! Next steps:"
echo ""
echo "1. Get a GitHub personal access token:"
echo "   https://github.com/settings/tokens/new"
echo "   - Select 'repo' scope"
echo "   - Copy the token"
echo ""
echo "2. Open Google Colab:"
echo "   https://colab.research.google.com"
echo ""
echo "3. Create a new notebook and:"
echo "   - Runtime → Change runtime type → GPU"
echo "   - Copy cells from notebooks/COLAB_PIPELINE.md"
echo ""
echo "4. Run the pipeline!"
echo "   - It will ask for your GitHub token"
echo "   - Everything else is automated"
echo ""
echo "📄 Reference: notebooks/COLAB_PIPELINE.md"
echo "📄 Instructions: notebooks/README.md"
