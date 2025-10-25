#!/bin/bash

# Monitor Dendritic Whisper Training
# Usage: ./monitor_training.sh [run_name]

RUN_NAME=${1:-test_run}
RESULTS_DIR="/Users/bledden/Documents/dendritic-hackathon/whisper-edge-optimization/results/$RUN_NAME"

echo "========================================"
echo "🔍 DENDRITIC TRAINING MONITOR"
echo "========================================"
echo "Run: $RUN_NAME"
echo "Results dir: $RESULTS_DIR"
echo ""

# Check if training is running
if ps aux | grep -q "[p]ython.*train_dendritic_full.py.*$RUN_NAME"; then
    echo "Status: 🟢 RUNNING"
else
    echo "Status: 🔴 NOT RUNNING"
fi
echo ""

# Show key metrics from results file if it exists
if [ -f "$RESULTS_DIR/final_results.json" ]; then
    echo "========================================"
    echo "📊 FINAL RESULTS"
    echo "========================================"
    cat "$RESULTS_DIR/final_results.json"
    echo ""
fi

# Look for PAI CSV files (created during training)
echo "========================================"
echo "📈 PAI TRACKING FILES"
echo "========================================"
PAI_DIR="/Users/bledden/Documents/dendritic-hackathon/PerforatedAI"
if [ -d "$PAI_DIR" ]; then
    echo "Recent PAI files:"
    ls -lth "$PAI_DIR"/*.csv 2>/dev/null | head -5 || echo "No CSV files yet"
    echo ""

    # Show parameter reduction from best test score file
    if [ -f "$PAI_DIR/bestTestScore.csv" ]; then
        echo "Parameter reduction history:"
        cat "$PAI_DIR/bestTestScore.csv" | tail -10
    fi
fi
echo ""

# Monitor for key events in real-time if no results yet
if [ ! -f "$RESULTS_DIR/final_results.json" ]; then
    echo "========================================"
    echo "🔄 LIVE MONITORING (Ctrl+C to stop)"
    echo "========================================"
    echo "Watching for key events..."
    echo ""

    # This will tail the Python process output
    # You can also redirect stdout to a log file in the training script
    echo "To see live output, run:"
    echo "  ps aux | grep 'train_dendritic_full.py'"
    echo ""
    echo "Or check recent output with:"
    echo "  tail -100 $RESULTS_DIR/*.log 2>/dev/null"
fi

echo "========================================"
echo "💡 USEFUL COMMANDS"
echo "========================================"
echo ""
echo "# Watch for dendrite additions:"
echo "  watch -n 5 'ps aux | grep train_dendritic'"
echo ""
echo "# Kill training if needed:"
echo "  pkill -f \"train_dendritic_full.py.*$RUN_NAME\""
echo ""
echo "# Check system resources:"
echo "  top -pid \$(pgrep -f \"train_dendritic_full.py.*$RUN_NAME\")"
echo ""
