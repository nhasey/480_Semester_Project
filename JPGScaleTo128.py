from PIL import Image
import os

def resize_images(input_folder, output_folder, max_size=(128, 128)):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Handle file extensions in a case-insensitive manner
            # Construct full file path
            file_path = os.path.join(input_folder, filename)

            # Open an image file
            with Image.open(file_path) as img:
                # Resize image while maintaining aspect ratio
                img.thumbnail(max_size)  # Remove the ANTIALIAS argument

                # Create a new image with a white background
                new_img = Image.new("RGB", max_size, (255, 255, 255))
                # Calculate position to paste the resized image onto the new image
                x = (max_size[0] - img.width) // 2
                y = (max_size[1] - img.height) // 2
                new_img.paste(img, (x, y))

                # Save resized image to output folder
                output_path = os.path.join(output_folder, filename)
                new_img.save(output_path)
                print(f'Resized and saved: {output_path}')

# Usage
input_folder = r"C:\Users\natha\OneDrive\Desktop\Aidan Test"
output_folder = r"C:\Users\natha\OneDrive\Desktop\Aidan Test\128"
resize_images(input_folder, output_folder)
