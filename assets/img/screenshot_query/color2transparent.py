from PIL import Image
import os

def color_distance(color1, color2):
    return sum((a - b) ** 2 for a, b in zip(color1, color2)) ** 0.5

def make_color_transparent(image_path, target_color, tolerance):
    img = Image.open(image_path)
    img = img.convert("RGBA")
    data = img.getdata()

    new_data = []
    for item in data:
        if color_distance(item[:3], target_color) <= tolerance:
            new_data.append((255, 255, 255, 0))  # Fully transparent
        else:
            new_data.append(item)

    img.putdata(new_data)
    return img

def process_folder(folder_path, target_color, tolerance):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.png'):
            file_path = os.path.join(folder_path, filename)
            new_img = make_color_transparent(file_path, target_color, tolerance)
            new_img.save(os.path.join(folder_path, f"transparent_{filename}"))

# Example usage
folder_path = "."  # Current directory
target_color = (0, 255, 0)  # RGB value of pure green
tolerance = 150  # Adjust this value to include more or fewer shades of green

process_folder(folder_path, target_color, tolerance)