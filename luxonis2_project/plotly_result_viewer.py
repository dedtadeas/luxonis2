import cv2
import plotly.graph_objs as go
import plotly.io as pio
import pandas as pd
from tqdm import tqdm


class PlotlyResultViewer:
    def __init__(
        self,
        tracked_objects: dict,
        video_path: str,
        output_file="output/tracking_results.html",
    ):
        self.tracked_objects = tracked_objects
        self.output_file = output_file
        self.frame_width, self.frame_height = self.get_video_dimensions(video_path)

    def get_video_dimensions(self, video_path):
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return width, height

    def generate_html(self, output_file=None):
        if output_file:
            self.output_file = output_file

        scatter_traces = []

        # Convert the tracked_objects dictionary to a DataFrame
        list_of_objects = []
        for object_id, obj_info in self.tracked_objects.items():
            for frame_id, coords in obj_info["path"]:
                list_of_objects.append(
                    [object_id, obj_info["shape"], obj_info["color"], frame_id, coords]
                )

        df_tracked = pd.DataFrame(
            list_of_objects,
            columns=["object_id", "shape", "color", "frame_id", "coords"],
        )

        # Group the tracked objects by object_id
        for object_id, group_df in tqdm(df_tracked.groupby("object_id")):
            group_df = group_df.sort_values(by="frame_id")
            x_coords = group_df["coords"].apply(lambda c: c[0])
            y_coords = group_df["coords"].apply(lambda c: c[1])

            # Convert numpy.uint8 color values to standard Python int
            color = tuple(int(c) for c in group_df["color"].iloc[0])
            color_str = f"rgb{color}"

            # Create a scatter trace for the object path, label only the first point
            scatter_trace = go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="lines+markers+text",
                name=f"Object {object_id}",
                line=dict(color=color_str),
                text=[
                    f"Object {object_id}" if i == 0 else ""
                    for i in range(len(x_coords))
                ],  # Label only the first point
                hovertext=[
                    f"Object {object_id} (Frame {frame_id})"
                    for frame_id in group_df["frame_id"]
                ],  # Hover text for all points
                hoverinfo="text",
                textposition="top center",
                visible=True,
            )
            scatter_traces.append(scatter_trace)

        # Create the figure with the scatter plot
        fig = go.Figure()

        # Add scatter plot traces
        for trace in scatter_traces:
            fig.add_trace(trace)

        # Adjust figure layout
        fig.update_layout(
            height=self.frame_height,
            width=self.frame_width, 
            hovermode="closest", 
            template="plotly_dark",
            xaxis=dict(range=[0, self.frame_width]), 
            yaxis=dict(range=[0, self.frame_height]), 
            xaxis_title="X Coordinate",
            yaxis_title="Y Coordinate",
        )

        # Generate the HTML content
        html_content = pio.to_html(fig, include_plotlyjs="cdn")

        # Save the HTML content to a file
        with open(self.output_file, "w") as f:
            f.write(html_content)

        print(f"HTML file saved to: {self.output_file}")