import cv2
import numpy as np

def create_outpainting_mask(image_path, target_size=(384, 384), buffer_ratio=0.1):
    # Load the image
    image = cv2.imread(image_path)

    # Calculate the buffer size based on the target size and buffer ratio
    buffer_size = int(min(target_size) * buffer_ratio)

    # Resize the image while maintaining aspect ratio
    height, width = image.shape[:2]
    aspect_ratio = width / height
    if aspect_ratio > 1:
        new_width = target_size[0] - (2 * buffer_size)
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = target_size[1] - (2 * buffer_size)
        new_width = int(new_height * aspect_ratio)

    resized_image = cv2.resize(image, (new_width, new_height))

    # Create a black mask of the target size
    mask = np.full(target_size, 255, dtype=np.uint8)

    # Determine the region of interest within the mask
    roi_start_x = int((target_size[1] - new_width) / 2)
    roi_start_y = int((target_size[0] - new_height) / 2)
    roi_end_x = roi_start_x + new_width
    roi_end_y = roi_start_y + new_height

    # Set the region of interest to white in the mask
    mask[roi_start_y:roi_end_y, roi_start_x:roi_end_x] = 0

    return mask, resized_image

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Invalid number of argument, requires 2.")

    input_image_path = sys.argv[1]
    mask = create_outpainting_mask(input_image_path)

    # Save the mask as an image
    cv2.imwrite(sys.argv[2], mask)