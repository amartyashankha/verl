#!/bin/bash

# Debug launcher for faster iteration during development
# Usage: ./debug_launch.sh

export DEBUG_MODE=true

echo "🐛 DEBUG MODE: Using faster settings for development"
echo "   - Smaller batch sizes"
echo "   - Fewer responses per prompt" 
echo "   - Skip initial validation"
echo "   - Use fewer tensor parallel workers"

# Launch with debug settings
./run_orchestrator_qwen2_14b.sh