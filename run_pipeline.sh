#!/bin/bash
# ==============================================================================
# Clinical Gait Analysis & Markerless Kinematics Pipeline Launcher
# ==============================================================================

if [ -f "keypoint/keypoint/bin/python" ]; then
    PYTHON_BIN="keypoint/keypoint/bin/python"
elif [ -f "/opt/miniconda3/envs/sam_3d_body/bin/python" ]; then
    PYTHON_BIN="/opt/miniconda3/envs/sam_3d_body/bin/python"
else
    PYTHON_BIN="python3"
fi

PORT=8080

echo "----------------------------------------------------------------------"
echo " Clinical Gait Analysis Tool: Normative Data Bank & Interventions "
echo "----------------------------------------------------------------------"

usage() {
    echo "Usage: ./run_pipeline.sh [OPTION]"
    echo "  --all            (Default) Build normative gait bank & launch clinical web dashboard"
    echo "  --gait-analysis  Generate normative data bank & run clinical decision support engine"
    echo "  --compare        Run comparative validation engine against Vicon C3D data"
    echo "  --dashboard      Launch interactive web dashboard server on http://localhost:$PORT"
    echo "  --help           Show this help message"
    exit 0
}

run_gait_analysis() {
    echo "📊 [1/2] Building Normative Gait Dataset Bank & Running Clinical Diagnostics..."
    $PYTHON_BIN normative_gait_data.py
    $PYTHON_BIN gait_analysis_engine.py
    echo "✅ Normative gait bank and clinical analysis complete! Payloads saved to App/"
}

run_compare() {
    echo "🚀 Running SAM 3D MHR vs Vicon Comparative Engine..."
    $PYTHON_BIN comparison_engine.py
    echo "✅ Validation complete! Generated figures in figures/"
}

run_dashboard() {
    if lsof -ti:$PORT >/dev/null 2>&1; then
        echo "⚠️ Port $PORT is currently in use. Clearing existing server process..."
        kill -9 $(lsof -ti:$PORT) 2>/dev/null || true
        sleep 1
    fi
    echo "🌐 Starting Interactive Clinical Web Dashboard on http://localhost:$PORT ..."
    echo "👉 Open http://localhost:$PORT in your web browser."
    echo "Press Ctrl+C to stop the server."
    $PYTHON_BIN -m http.server $PORT --directory App
}

case "$1" in
    --gait-analysis)
        run_gait_analysis
        ;;
    --compare)
        run_compare
        ;;
    --dashboard)
        run_dashboard
        ;;
    --help|-h)
        usage
        ;;
    --all|"")
        run_gait_analysis
        echo ""
        run_dashboard
        ;;
    *)
        echo "Unknown option: $1"
        usage
        ;;
esac
