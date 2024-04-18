import os
import random
from PIL import Image

# Function to combine four images into one
def combine_images(images):
    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    total_height = sum(heights)

    new_img = Image.new('RGB', (max_width, total_height))

    y_offset = 0
    for img in images:
        new_img.paste(img, (0, y_offset))
        y_offset += img.size[1]

    return new_img

# Path to the folder containing the shape images
dataset_path = "../benchmark/datasets/geometric shapes dataset"
output_path = "../benchmark/datasets/shapes"

# Create output directory if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

# List of shape categories
shape_categories = ["Circle", "Square", "Triangle"]

# Number of combined images to generate
num_images = 1000

# Loop to generate combined images
for i in range(num_images):
    combined_image = []
    for j in range(4):
        # Randomly select a shape category
        category = random.choice(shape_categories)
        # List images in the selected category
        shape_images = os.listdir(os.path.join(dataset_path, category))
        # Randomly select an image from the category
        image_file = random.choice(shape_images)
        # Open and append the selected image
        img = Image.open(os.path.join(dataset_path, category, image_file))
        combined_image.append(img)

    # Combine the selected images into one
    new_image = combine_images(combined_image)
    # Save the combined image
    new_image.save(os.path.join(output_path, f"combined_{i}.png"))

print("Combined images generated successfully!")