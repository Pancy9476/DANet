import os
import cv2

def convert_rgb_images_to_gray(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            # Construct the input and output paths
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"{filename}")

            # Read the RGB image
            rgb_image = cv2.imread(input_path)

            if rgb_image is not None:
                # Convert to grayscale
                gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

                # Save the grayscale image
                cv2.imwrite(output_path, gray_image)
                print(f"Gray image saved to {output_path}")
            else:
                print(f"Failed to read the RGB image from {input_path}")

# Example usage
input_folder_path = "test_img/LLVIP/ir"
output_folder_path = "single_chanel/LLVIP/ir"
convert_rgb_images_to_gray(input_folder_path, output_folder_path)
