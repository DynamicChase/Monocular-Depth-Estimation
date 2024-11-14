import os
import numpy as np
import cv2
import torch
from torchvision import transforms
import matplotlib as mpl
import matplotlib.cm as cm
from PIL import Image as pil
import time

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist

# Define the function to load the model
def load_model(model_name, device):
    download_model_if_doesnt_exist(model_name)
    model_path = os.path.join("models", model_name)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # Get input size from model
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)
    depth_decoder.to(device)
    depth_decoder.eval()

    return encoder, depth_decoder, feed_width, feed_height

# Define the function to process and display the depth map from webcam
def process_webcam(model_name="mono+stereo_640x192", device="cuda" if torch.cuda.is_available() else "cpu"):
    # Load model
    encoder, depth_decoder, feed_width, feed_height = load_model(model_name, device)

    # Start video capture
    cap = cv2.VideoCapture(0) # '0' for the default webcam

    with torch.no_grad():
        while cap.isOpened():
            start_time = time.time()  # Start timing

            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to PIL image and resize
            input_image = pil.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0).to(device)

            # Run the frame through the model
            features = encoder(input_image)
            outputs = depth_decoder(features)
            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(disp, (frame.shape[0], frame.shape[1]), mode="bilinear", align_corners=False)
            disp_resized_np = disp_resized.squeeze().cpu().numpy()

            # Normalize and colormap the disparity map for display
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)

            # Display the depth map
            depth_map = cv2.cvtColor(colormapped_im, cv2.COLOR_RGB2BGR)

            # Calculate and display FPS and processing time
            end_time = time.time()
            processing_time = end_time - start_time
            fps = 1.0 / processing_time

            # Add FPS and processing time to the frame
            cv2.putText(depth_map, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(depth_map, f"Time: {processing_time:.3f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show the depth map with FPS
            cv2.imshow("Monocular Depth Estimation", depth_map)
            cv2.imshow("colored image", frame)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_webcam()

