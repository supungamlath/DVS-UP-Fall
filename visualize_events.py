import h5py
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt


def generate_sample_dvs_events(n_events=1000, width=64, height=64, duration=1.0):
    """
    Generate sample DVS events for demonstration.
    Args:
        n_events: Number of events to generate
        width: Sensor width in pixels
        height: Sensor height in pixels
        duration: Duration in seconds
    Returns:
        List of tuples (timestamp, x, y, polarity)
    """
    # Generate random events
    timestamps = np.sort(np.random.uniform(0, duration, n_events))
    x_coords = np.random.randint(0, width, n_events)
    y_coords = np.random.randint(0, height, n_events)
    polarities = np.random.choice([0, 1], n_events)  # 0 for negative, 1 for positive
    events = [(t, x, y, p) for t, x, y, p in zip(timestamps, x_coords, y_coords, polarities)]
    return events


def visualize_dvs_events_3d(spike_tuples, title="DVS Events 3D Visualization", sensor_height=None):
    """
    Visualize DVS events in 3D using Plotly with correct coordinate system.
    Args:
        spike_tuples: numpy array with shape (n_events, 4) containing [timestamp, x, y, polarity]
        title: Title for the plot
        sensor_height: Height of the sensor in pixels (for coordinate correction). If None, auto-detect from data.
    """
    timestamps = spike_tuples[:, 0] / 1e6  # Convert to seconds
    x_coords = spike_tuples[:, 1]
    y_coords = spike_tuples[:, 2]
    polarities = spike_tuples[:, 3]

    # Auto-detect sensor height if not provided
    if sensor_height is None:
        sensor_height = int(np.max(y_coords)) + 1

    # Flip Y coordinates to match DVS coordinate system (top-left origin)
    y_coords_flipped = sensor_height - 1 - y_coords

    # Separate events by polarity
    pos_mask = polarities == 1
    neg_mask = polarities == 0

    # Create the 3D scatter plot
    fig = go.Figure()

    # Add positive polarity events (red)
    if np.any(pos_mask):
        fig.add_trace(
            go.Scatter3d(
                x=timestamps[pos_mask],  # Time on X-axis
                y=x_coords[pos_mask],  # X coordinates on Y-axis
                z=y_coords_flipped[pos_mask],  # Flipped Y coordinates on Z-axis
                mode="markers",
                marker=dict(color="red", size=3, opacity=0.7),
                name="Positive Events",
                hovertemplate="<b>Positive Event</b><br>"
                + "Time: %{x:.4f}s<br>"
                + "X: %{y}<br>"
                + "Y: %{z}<br>"
                + "<extra></extra>",
            )
        )

    # Add negative polarity events (blue)
    if np.any(neg_mask):
        fig.add_trace(
            go.Scatter3d(
                x=timestamps[neg_mask],  # Time on X-axis
                y=x_coords[neg_mask],  # X coordinates on Y-axis
                z=y_coords_flipped[neg_mask],  # Flipped Y coordinates on Z-axis
                mode="markers",
                marker=dict(color="blue", size=3, opacity=0.7),
                name="Negative Events",
                hovertemplate="<b>Negative Event</b><br>"
                + "Time: %{x:.4f}s<br>"
                + "X: %{y}<br>"
                + "Y: %{z}<br>"
                + "<extra></extra>",
            )
        )

    # Update layout with corrected axis labels
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        scene=dict(
            xaxis_title="Time (seconds)",
            yaxis_title="X Coordinate (pixels)",
            zaxis_title="Y Coordinate (pixels)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            aspectmode="manual",
            aspectratio=dict(x=10, y=1, z=1),
            # Reverse Z-axis to match DVS coordinate system
            zaxis=dict(autorange="reversed"),
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    return fig


def create_time_slice_animation(spike_tuples, n_slices=50, title="DVS Events Animation", sensor_height=None):
    """
    Create an animated visualization showing events over time slices with correct coordinate system.
    Args:
        spike_tuples: numpy array with shape (n_events, 4) containing [timestamp, x, y, polarity]
        n_slices: Number of time slices for animation
        title: Title for the plot
        sensor_height: Height of the sensor in pixels (for coordinate correction). If None, auto-detect from data.
    """
    # Extract data
    timestamps = spike_tuples[:, 0] / 1e6  # Convert to seconds
    x_coords = spike_tuples[:, 1]
    y_coords = spike_tuples[:, 2]
    polarities = spike_tuples[:, 3]

    # Auto-detect sensor height if not provided
    if sensor_height is None:
        sensor_height = int(np.max(y_coords)) + 1

    # Flip Y coordinates to match DVS coordinate system
    y_coords_flipped = sensor_height - 1 - y_coords

    # Create time slices
    min_time, max_time = timestamps.min(), timestamps.max()
    time_slices = np.linspace(min_time, max_time, n_slices)

    frames = []
    for i, t in enumerate(time_slices):
        # Get events up to current time
        mask = timestamps <= t
        current_timestamps = timestamps[mask]
        current_x = x_coords[mask]
        current_y_flipped = y_coords_flipped[mask]
        current_pol = polarities[mask]

        # Separate by polarity
        pos_mask = current_pol == 1
        neg_mask = current_pol == 0

        frame_data = []

        # Positive events
        if np.any(pos_mask):
            frame_data.append(
                go.Scatter3d(
                    x=current_timestamps[pos_mask],
                    y=current_x[pos_mask],
                    z=current_y_flipped[pos_mask],
                    mode="markers",
                    marker=dict(color="red", size=3, opacity=0.7),
                    name="Positive Events",
                )
            )

        # Negative events
        if np.any(neg_mask):
            frame_data.append(
                go.Scatter3d(
                    x=current_timestamps[neg_mask],
                    y=current_x[neg_mask],
                    z=current_y_flipped[neg_mask],
                    mode="markers",
                    marker=dict(color="blue", size=3, opacity=0.7),
                    name="Negative Events",
                )
            )

        frames.append(
            go.Frame(data=frame_data, name=f"frame_{i}", layout=go.Layout(title_text=f"{title} - Time: {t:.3f}s"))
        )

    # Create initial figure
    fig = go.Figure(data=frames[0].data if frames else [], frames=frames)

    # Add animation controls
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Time (seconds)",
            yaxis_title="X Coordinate (pixels)",
            zaxis_title="Y Coordinate (pixels)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            # Reverse Z-axis to match DVS coordinate system
            zaxis=dict(autorange="reversed"),
        ),
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                steps=[
                    dict(
                        args=[[f"frame_{i}"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                        label=f"{time_slices[i]:.3f}s",
                        method="animate",
                    )
                    for i in range(len(time_slices))
                ],
                active=0,
                currentvalue={"prefix": "Time: "},
                len=0.9,
                x=0.1,
                y=0,
                xanchor="left",
                yanchor="top",
            )
        ],
    )

    return fig


def visualize_2d_snapshot(spike_tuples, time_window=None, title="DVS Events 2D View"):
    """
    Create a 2D visualization of events at a specific time or time window.
    Args:
        spike_tuples: numpy array with shape (n_events, 4) containing [timestamp, x, y, polarity]
        time_window: tuple (start_time, end_time) in seconds. If None, shows all events.
        title: Title for the plot
        sensor_height: Height of the sensor in pixels. If None, auto-detect from data.
    """
    timestamps = spike_tuples[:, 0] / 1e6
    x_coords = spike_tuples[:, 1]
    y_coords = spike_tuples[:, 2]
    polarities = spike_tuples[:, 3]

    sensor_height = 120

    # Filter by time window if specified
    if time_window is not None:
        start_time, end_time = time_window
        mask = (timestamps >= start_time) & (timestamps <= end_time)
        timestamps = timestamps[mask]
        x_coords = x_coords[mask]
        y_coords = y_coords[mask]
        polarities = polarities[mask]

    # Flip Y coordinates
    y_coords_flipped = sensor_height - 1 - y_coords

    # Separate by polarity
    pos_mask = polarities == 1
    neg_mask = polarities == 0

    fig = go.Figure()

    # Add positive events
    if np.any(pos_mask):
        fig.add_trace(
            go.Scatter(
                x=x_coords[pos_mask],
                y=y_coords_flipped[pos_mask],
                mode="markers",
                marker=dict(color="red", size=4, opacity=0.7),
                name="Positive Events",
            )
        )

    # Add negative events
    if np.any(neg_mask):
        fig.add_trace(
            go.Scatter(
                x=x_coords[neg_mask],
                y=y_coords_flipped[neg_mask],
                mode="markers",
                marker=dict(color="blue", size=4, opacity=0.7),
                name="Negative Events",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="X Coordinate (pixels)",
        yaxis_title="Y Coordinate (pixels)",
        yaxis=dict(scaleanchor="x", scaleratio=1),  # Equal aspect ratio
    )

    return fig


def draw_2d_snapshot(spike_tuples, time_window=None, title="DVS Events 2D View"):
    """
    Create a 2D visualization of events at a specific time or time window using matplotlib.

    Args:
        spike_tuples: numpy array with shape (n_events, 4) containing [timestamp, x, y, polarity]
        time_window: tuple (start_time, end_time) in seconds. If None, shows all events.
        title: Title for the plot

    Returns:
        fig, ax: matplotlib figure and axis objects
    """
    timestamps = spike_tuples[:, 0] / 1e6
    x_coords = spike_tuples[:, 1].astype(int)
    y_coords = spike_tuples[:, 2].astype(int)
    polarities = spike_tuples[:, 3]

    # Fixed sensor dimensions
    sensor_width = 160
    sensor_height = 120

    # Filter by time window if specified
    if time_window is not None:
        start_time, end_time = time_window
        mask = (timestamps >= start_time) & (timestamps <= end_time)
        timestamps = timestamps[mask]
        x_coords = x_coords[mask]
        y_coords = y_coords[mask]
        polarities = polarities[mask]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create accumulated event frame
    frame = np.zeros((sensor_height, sensor_width))

    # Ensure coordinates are within bounds
    valid_mask = (x_coords >= 0) & (x_coords < sensor_width) & (y_coords >= 0) & (y_coords < sensor_height)
    x_coords = x_coords[valid_mask]
    y_coords = y_coords[valid_mask]
    polarities = polarities[valid_mask]

    # Accumulate events (positive events add +1, negative events add -1)
    for x, y, pol in zip(x_coords, y_coords, polarities):
        frame[y, x] += 1 if pol == 1 else -1

    # Display the frame
    im = ax.imshow(
        frame,
        cmap="RdBu_r",
        origin="upper",
        vmin=-np.max(np.abs(frame)),
        vmax=np.max(np.abs(frame)),
        extent=[0, sensor_width, sensor_height, 0],
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Event Count (Red: Positive, Blue: Negative)")

    ax.set_xlabel("X Coordinate (pixels)")
    ax.set_ylabel("Y Coordinate (pixels)")
    ax.set_title(title)
    ax.set_aspect("equal")

    # Add grid for better visualization
    ax.grid(True, alpha=0.3)

    # Add event count information to title
    if time_window is not None:
        event_count = len(x_coords)
        time_info = f" (t={time_window[0]:.3f}-{time_window[1]:.3f}s, {event_count} events)"
        ax.set_title(title + time_info)

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Load your DVS events
    v2e_events_file_path = ""
    v2ce_events_file_path = ""

    with h5py.File(v2e_events_file_path, "r") as spike_data:
        spike_array = np.array(spike_data["events"])

        print(f"Loaded {len(spike_array)} events")
        print(f"Data shape: {spike_array.shape}")
        print(f"Time range: {spike_array[:, 0].min()/1e6:.3f} - {spike_array[:, 0].max()/1e6:.3f} seconds")
        print(f"Spatial range: X(0-{spike_array[:, 1].max()}), Y(0-{spike_array[:, 2].max()})")

        # Create 3D visualization with corrected coordinates
        # fig_3d = visualize_dvs_events_3d(spike_tuples, "DVS Events 3D - Corrected Coordinates")
        # fig_3d.show()

        # # Create 2D snapshot for comparison
        # fig_2d = visualize_2d_snapshot(spike_tuples, time_window=(0, 1.0), title="DVS Events 2D View (First 100ms)")

        # draw_2d_snapshot(spike_tuples, time_window=(0, 1.0), title="DVS Events 2D View (First 1s)")

        # Uncomment to create animation
        # fig_anim = create_time_slice_animation(spike_tuples, n_slices=30)
        # fig_anim.show()

    spike_data = np.load(v2ce_events_file_path)
    spike_array = np.array([list(spike) for spike in spike_data["event_stream"]])

    print(f"Loaded {len(spike_array)} events")
    print(f"Data shape: {spike_array.shape}")
    print(f"Time range: {spike_array[:, 0].min()/1e6:.3f} - {spike_array[:, 0].max()/1e6:.3f} seconds")
    print(f"Spatial range: X(0-{spike_array[:, 1].max()}), Y(0-{spike_array[:, 2].max()})")

    # Create 2D snapshot for comparison
    draw_2d_snapshot(spike_array, time_window=(0, 1.0), title="DVS Events 2D View (First 1s)")
