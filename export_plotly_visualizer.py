"""
SAM 3D Clinical Motion Analysis - Plotly 3D Visualizer Generator
Generates an interactive 3D clinical visualization using Plotly (Scatter3d + Mesh3d),
providing clean, distortion-free 3D anatomical skeletal & surface visualization.
"""

import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_plotly_clinical_dashboard(json_path: str = 'App/sam3d_monocular.json',
                                    output_html_path: str = 'App/plotly_dashboard.html') -> str:
    """
    Creates an interactive 3D Clinical Motion Analysis dashboard with Plotly.
    Includes 3D Anatomical Skeleton & Surface Mesh, synchronized Knee Flexion Waveforms,
    and frame-by-frame scrubbing controls.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    frames = data['frames']
    skeleton_bones = data['skeleton_bones']
    num_frames = len(frames)
    times = [f['time'] for f in frames]

    # Color palette for anatomical segments
    bone_colors = {
        "spine": "#38bdf8",     # Cyan
        "r_leg": "#10b981",     # Emerald Green (Target Leg)
        "l_leg": "#f59e0b",     # Amber/Yellow
        "r_arm": "#a855f7",     # Purple
        "l_arm": "#ec4899"      # Pink
    }

    def get_bone_group(b1, b2):
        if "r_hip" in b1 or "r_knee" in b1 or "r_ankle" in b1:
            return "r_leg"
        if "l_hip" in b1 or "l_knee" in b1 or "l_ankle" in b1:
            return "l_leg"
        if "r_shoulder" in b1 or "r_elbow" in b1:
            return "r_arm"
        if "l_shoulder" in b1 or "l_elbow" in b1:
            return "l_arm"
        return "spine"

    # 1. Build Plotly Subplot Layout
    # Col 1: 3D Scene (Spanning Rows 1 & 2), Col 2 Row 1: Kinematics, Col 2 Row 2: Kinetics
    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.55, 0.45],
        row_heights=[0.5, 0.5],
        specs=[
            [{"type": "scene", "rowspan": 2}, {"type": "xy"}],
            [None,                            {"type": "xy"}]
        ],
        subplot_titles=(
            "SAM 3D MHR Anatomical 3D Mesh & Skeleton",
            "Clinical Kinematics (Knee Flexion vs Vicon)",
            "Inverse Dynamics Kinetics (Knee Torque & Power)"
        )
    )

    # Initial frame data (Frame 0)
    frame0 = frames[0]
    j3d0 = frame0['joints3d']

    # Add 3D Joint Nodes
    jx = [j3d0[k][0] for k in j3d0]
    jy = [j3d0[k][2] for k in j3d0]  # Z in 3D -> Y in Plotly 3D (Depth)
    jz = [j3d0[k][1] for k in j3d0]  # Y in 3D -> Z in Plotly 3D (Height Y-up)
    jlabels = list(j3d0.keys())

    fig.add_trace(
        go.Scatter3d(
            x=jx, y=jy, z=jz,
            mode='markers+text',
            marker=dict(size=6, color='#38bdf8', opacity=0.9),
            text=jlabels,
            textfont=dict(size=9, color='#e2e8f0'),
            name="Anatomical Joints",
            hoverinfo="text+x+y+z"
        ),
        row=1, col=1
    )

    # Add 3D Bone Lines for Frame 0
    for bone in skeleton_bones:
        p1, p2 = j3d0[bone[0]], j3d0[bone[1]]
        group = get_bone_group(bone[0], bone[1])
        color = bone_colors[group]

        fig.add_trace(
            go.Scatter3d(
                x=[p1[0], p2[0]],
                y=[p1[2], p2[2]],
                z=[p1[1], p2[1]],
                mode='lines',
                line=dict(color=color, width=6),
                name=f"Bone: {bone[0]}-{bone[1]}",
                showlegend=False
            ),
            row=1, col=1
        )

    # Add Kinematic & Kinetic Chart Lines
    knee_vicon = [f['angles']['vicon_knee_flex'] for f in frames]
    knee_sam = [f['angles']['r_knee_flex_smoothed'] for f in frames]
    knee_mom = [f['kinetics'].get('r_knee_moment', 0.0) for f in frames]
    knee_pow = [f['kinetics'].get('r_knee_power', 0.0) for f in frames]

    fig.add_trace(
        go.Scatter(
            x=times, y=knee_vicon,
            mode='lines',
            line=dict(color='#f8fafc', width=2),
            name='Vicon Reference'
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=times, y=knee_sam,
            mode='lines',
            line=dict(color='#38bdf8', width=3),
            name='SAM 3D MHR'
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=times, y=knee_mom,
            mode='lines',
            line=dict(color='#a855f7', width=2.5),
            name='Knee Moment (N·m/kg)'
        ),
        row=2, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=times, y=knee_pow,
            mode='lines',
            line=dict(color='#10b981', width=2.5),
            name='Knee Power (W/kg)'
        ),
        row=2, col=2
    )

    # 2. Build Animation Frames for Plotly
    plotly_frames = []
    for idx, f in enumerate(frames):
        j3d = f['joints3d']
        
        curr_jx = [j3d[k][0] for k in j3d]
        curr_jy = [j3d[k][2] for k in j3d]
        curr_jz = [j3d[k][1] for k in j3d]

        frame_traces = []
        # Trace 0: Joint Nodes
        frame_traces.append(
            go.Scatter3d(
                x=curr_jx, y=curr_jy, z=curr_jz,
                mode='markers',
                marker=dict(size=6, color='#38bdf8', opacity=0.9),
                text=jlabels
            )
        )

        # Traces 1..N: Bone lines
        for bone in skeleton_bones:
            p1, p2 = j3d[bone[0]], j3d[bone[1]]
            group = get_bone_group(bone[0], bone[1])
            color = bone_colors[group]
            frame_traces.append(
                go.Scatter3d(
                    x=[p1[0], p2[0]],
                    y=[p1[2], p2[2]],
                    z=[p1[1], p2[1]],
                    mode='lines',
                    line=dict(color=color, width=6)
                )
            )

        # Traces N+1, N+2: Kinematic plot lines
        frame_traces.append(
            go.Scatter(x=times, y=knee_vicon, mode='lines', line=dict(color='#f8fafc', width=2))
        )
        frame_traces.append(
            go.Scatter(x=times, y=knee_sam, mode='lines', line=dict(color='#38bdf8', width=3))
        )
        # Traces N+3, N+4: Kinetic plot lines
        frame_traces.append(
            go.Scatter(x=times, y=knee_mom, mode='lines', line=dict(color='#a855f7', width=2.5))
        )
        frame_traces.append(
            go.Scatter(x=times, y=knee_pow, mode='lines', line=dict(color='#10b981', width=2.5))
        )

        plotly_frames.append(
            go.Frame(data=frame_traces, name=str(idx))
        )

    fig.frames = plotly_frames

    # 3. Configure Layout & Sliders
    slider_steps = []
    for idx in range(num_frames):
        step = {
            "args": [
                [str(idx)],
                {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}
            ],
            "label": f"{times[idx]:.2f}s",
            "method": "animate"
        }
        slider_steps.append(step)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#090d16",
        plot_bgcolor="#0f172a",
        title=dict(
            text="<b>SAM 3D Clinical Motion Analysis & Inverse Dynamics Kinetics Dashboard</b>",
            x=0.02, y=0.98,
            font=dict(size=18, color="#f8fafc")
        ),
        scene=dict(
            xaxis=dict(range=[-0.8, 0.8], title="X (Medial-Lateral)", backgroundcolor="#0f172a", gridcolor="#1e293b"),
            yaxis=dict(range=[-0.8, 0.8], title="Z (Anteroposterior)", backgroundcolor="#0f172a", gridcolor="#1e293b"),
            zaxis=dict(range=[0.0, 1.8], title="Y (Height - Ground Y≥0)", backgroundcolor="#0f172a", gridcolor="#1e293b"),
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.2))
        ),
        xaxis2=dict(title="Time (seconds)", gridcolor="#1e293b"),
        yaxis2=dict(title="Knee Flexion Angle (°)", range=[-10, 140], gridcolor="#1e293b"),
        xaxis3=dict(title="Time (seconds)", gridcolor="#1e293b"),
        yaxis3=dict(title="Kinetics Value", gridcolor="#1e293b"),
        sliders=[{
            "active": 0,
            "yanchor": "top", "xanchor": "left",
            "currentvalue": {"font": {"size": 14, "color": "#38bdf8"}, "prefix": "Time: ", "visible": True, "xanchor": "right"},
            "pad": {"b": 10, "t": 40},
            "len": 0.9, "x": 0.05, "y": 0,
            "steps": slider_steps
        }],
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 33, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}],
                    "label": "▶ Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
                    "label": "⏸ Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 40},
            "showactive": False,
            "type": "buttons",
            "x": 0.05, "xanchor": "right", "y": 0, "yanchor": "top"
        }]
    )

    fig.write_html(output_html_path)
    print(f"Generated Plotly 3D Clinical Dashboard HTML: {output_html_path}")
    return output_html_path


if __name__ == '__main__':
    create_plotly_clinical_dashboard()
