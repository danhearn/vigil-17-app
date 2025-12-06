import os
from pathlib import Path
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import numpy as np
import hailo
import cv2
from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.depth.depth_pipeline import GStreamerDepthApp

# User-defined class to be used in the callback function: Inheritance from the app_callback_class
class user_app_callback_class(app_callback_class):

    def __init__(self):
        super().__init__()

    def calculate_average_depth(self, depth_mat):
        depth_values = np.array(depth_mat).flatten()  # Flatten the array and filter out outlier pixels
        try:
            m_depth_values = depth_values[depth_values <= np.percentile(depth_values, 95)]  # drop 5% of highest values (outliers)          
        except Exception as e:
            m_depth_values = np.array([])
        if len(m_depth_values) > 0:
            average_depth = np.mean(m_depth_values)  # Calculate the average depth of the pixels
        else:
            average_depth = 0  # Default value if no valid pixels are found
        return average_depth

# User-defined callback function: This is the callback function that will be called when data is available from the pipeline
def app_callback(pad, info, user_data):
    
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    if buffer is None: 
        return Gst.PadProbeReturn.OK
    
    # Using the user_data to count the number of frame
    user_data.increment()
    count = user_data.get_count()
    string_to_print = f"Frame count: {count}\n"

    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)
    frame = None

    # If use_frame flag is set to True
    if user_data.use_frame and format is not None and width is not None and height is not None:
        # Get video frame
        frame = get_numpy_from_buffer(buffer, format, width, height)
        print("Frame obtained")
        cv2.imwrite(f"frames/frame_{count}.jpg", frame)
        # frame = 255 - frame
        # pil_img = Image.fromarray(frame).convert("L")
        # draw = ImageDraw.Draw(pil_img)
        #frame = np.array(pil_img)

    roi = hailo.get_roi_from_buffer(buffer)
    depth_mat = roi.get_objects_typed(hailo.HAILO_DEPTH_MASK)
    depth_mat = depth_mat[0]
    depth_mat = depth_mat.get_data()
    depth_mat = np.array(depth_mat).reshape((256, 320))
    print(depth_mat[10][10])

    # if len(depth_mat) > 0:
    #     detection_average_depth = user_data.calculate_average_depth(depth_mat[0].get_data())
    # else:
    #     detection_average_depth = 0
    # string_to_print += (f"average depth: {detection_average_depth:.2f}\n")
    # print(string_to_print)
    depth_norm = cv2.normalize(depth_mat, None, 0, 255, cv2.NORM_MINMAX)
    depth_norm = depth_norm.astype(np.uint8)
    depth_colour = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    output_path = f"frames/depth_colormap_{count}.jpg"
    cv2.imwrite(output_path, depth_colour)
    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    env_file     = project_root / ".env"
    env_path_str = str(env_file)
    os.environ["HAILO_ENV_FILE"] = env_path_str

    user_data = user_app_callback_class()
    app = GStreamerDepthApp(app_callback, user_data)
    app.run()