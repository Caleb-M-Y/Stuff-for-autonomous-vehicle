import cv2
import numpy as np
from hailo_platform import HEF, VDevice, ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType

# The absolute law: This must match your YAML file exactly.
CLASS_NAMES = [
    "blue_ball", "blue_bucket", "green_ball", "green_bucket",
    "red_ball", "red_bucket", "yellow_ball", "yellow_bucket",
]

def run_hailo_inference(hef_path, image_path):
    # 1. Load the HEF and Image
    print("Loading HEF...")
    hef = HEF(hef_path)
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not find {image_path}")
        return

    # 2. Get the model's expected input size (usually 640x640)
    input_info = hef.get_input_vstream_infos()[0]
    height, width = input_info.shape[1], input_info.shape[2]
    
    # 3. Format image for the Hailo chip (Resize -> RGB -> Expand to batch shape)
    img_resized = cv2.resize(img, (width, height))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(img_rgb, axis=0)

    print("Connecting to physical Hailo NPU...")
    # 4. Connect to the chip and allocate memory
    with VDevice() as target:
        configure_params = ConfigureParams.create_from_hef(hef, interface=target.create_default_stream_interface())
        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()

        input_vstreams_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
        output_vstreams_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

        # 5. Run the actual inference
        with network_group.ActivatedNetworkGroup(network_group_params):
            with network_group.infer_context(input_vstreams_params, output_vstreams_params) as context:
                
                print("Sending image to NPU...")
                context.input_vstreams[input_info.name].send(input_data)
                
                print("Receiving bounding boxes from NPU...")
                output_info = hef.get_output_vstream_infos()[0]
                raw_output = context.output_vstreams[output_info.name].recv()
                
                # Hailo NMS outputs an array of [ymin, xmin, ymax, xmax, confidence, class_id]
                detections = raw_output[0] 
                
                for det in detections:
                    ymin, xmin, ymax, xmax, conf, cls_id = det
                    
                    if conf > 0.40: # 40% Confidence threshold
                        # Convert normalized Hailo coordinates back to your original image size
                        x1 = int(xmin * img.shape[1])
                        y1 = int(ymin * img.shape[0])
                        x2 = int(xmax * img.shape[1])
                        y2 = int(ymax * img.shape[0])
                        
                        class_id = int(cls_id)
                        label = f"{CLASS_NAMES[class_id]} {conf:.2f}"
                        print(f"SUCCESS: Found {label}!")
                        
                        # Draw the box and text
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 6. Save the verified image!
    cv2.imwrite("hailo_verified_output.jpg", img)
    print("Saved hailo_verified_output.jpg to folder. Go look at it!")

if __name__ == "__main__":
    # Ensure your HEF and image are in the same folder as this script on the robot
    run_hailo_inference("yolov8l.hef", "test_image.jpg")