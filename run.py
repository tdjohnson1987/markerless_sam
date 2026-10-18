"""
SAM 3D Kinematics & Kinetics Pipeline Python CLI Launcher
"""

import argparse
import subprocess
import sys
from comparison_engine import align_and_compare


def main():
    parser = argparse.ArgumentParser(description="SAM 3D Markerless Kinematics & Kinetics Pipeline")
    parser.add_argument("--vicon", type=str, default="vicon/lunge/s1_lunge_0_1.c3d", help="Path to Vicon C3D file")
    parser.add_argument("--sam3d", type=str, default="kinematics_data_sam.csv", help="Path to SAM 3D MHR kinematics CSV file")
    parser.add_argument("--output", type=str, default="figures", help="Output directory for figures & summary JSON")
    parser.add_argument("--serve", action="store_true", help="Launch interactive web dashboard server after validation")
    parser.add_argument("--port", type=int, default=8080, help="Web server port (default: 8080)")

    args = parser.parse_args()

    print(f"🚀 Running SAM 3D MHR vs Vicon validation...")
    print(f"   Vicon C3D Path: {args.vicon}")
    print(f"   SAM 3D CSV Path: {args.sam3d}")
    print(f"   Output Directory: {args.output}")

    metrics = align_and_compare(args.vicon, args.sam3d, args.output)

    print("\n✅ Validation Complete!")
    print(f"   Knee Flexion Pearson r:  {metrics['knee_raw']['pearson_r']}")
    print(f"   Knee Flexion MAE:        {metrics['knee_raw']['mae']}°")
    print(f"   Hip Flexion Pearson r:   {metrics['hip_smoothed']['pearson_r']}")
    print(f"   Hip Flexion MAE:         {metrics['hip_smoothed']['mae']}°")

    if args.serve:
        print(f"\n🌐 Starting web dashboard on http://localhost:{args.port}...")
        try:
            subprocess.run([sys.executable, "-m", "http.server", str(args.port), "--directory", "App"])
        except KeyboardInterrupt:
            print("\nStopped dashboard server.")


if __name__ == "__main__":
    main()
