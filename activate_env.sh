#!/bin/bash
# BSV Repository Prioritizer - Virtual Environment Activation

echo "Activating BSV Repository Prioritizer virtual environment..."

# Activate virtual environment
source venv/bin/activate

# Check if activation was successful
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment activated successfully!"
    echo "💡 Virtual environment: $VIRTUAL_ENV"
    echo ""
    echo "Available commands:"
    echo "  python src/test_collection.py          # Test data collection"
    echo "  python src/data_collection_runner.py   # Run full data collection"
    echo "  deactivate                             # Exit virtual environment"
    echo ""
else
    echo "❌ Failed to activate virtual environment"
fi
