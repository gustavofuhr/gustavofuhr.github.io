import os
from PIL import Image

def crop_images(folder_path, tl, br):
    # Ensure the output folder exists
    # output_folder = os.path.join(folder_path, "cropped")
    # os.makedirs(output_folder, exist_ok=True)

    # Get all image files in the folder
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]

    for image_file in image_files:
        # Open the image
        img_path = os.path.join(folder_path, image_file)
        with Image.open(img_path) as img:
            # Crop the image
            cropped_img = img.crop((tl[0], tl[1], br[0], br[1]))
            
            # Save the cropped image
            output_path = os.path.join(folder_path, f"cropped_{image_file}")
            cropped_img.save(output_path)
            print(f"Cropped and saved: {output_path}")

# Example usage
folder_path = "sbert/"
top_left = (280, 290)  # (x, y) coordinates of the top-left corner
bottom_right = (3311, 2266)  # (x, y) coordinates of the bottom-right corner

crop_images(folder_path, top_left, bottom_right)