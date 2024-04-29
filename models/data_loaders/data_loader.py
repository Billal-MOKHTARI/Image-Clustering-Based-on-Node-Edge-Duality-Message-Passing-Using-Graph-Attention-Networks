from torchvision import transforms # Custom dataset class to load images without labels
from PIL import Image
import torch
import pandas as pd
import numpy as np
import random
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src import utils

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
# Function to combine two images horizontally
def combine_horizontally(image1, image2):
    width1, height1 = image1.size
    width2, height2 = image2.size
    total_width = width1 + width2
    max_height = max(height1, height2)

    new_img = Image.new('RGB', (total_width, max_height))
    new_img.paste(image1, (0, 0))
    new_img.paste(image2, (width1, 0))

    return new_img

# Function to create combined images
def create_combined_images(dataset_path, output_path, rows = 2, cols = 2, num_images=1000):  
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path) 
    # Number of combined images to generate
    categories = []
    contents = os.listdir(dataset_path)
    subfolders = [f for f in contents if os.path.isdir(os.path.join(dataset_path, f))]

    # Print the basenames of the subfolders
    for subfolder in subfolders:
        categories.append(os.path.basename(subfolder))

    # Loop to generate combined images
    for i in range(num_images):
        combined_image = []
        for j in range(rows):  # Two images per row
            row_images = []
            for _ in range(cols):  # Two images per row
                # Randomly select a shape category
                category = random.choice(categories)
                # List images in the selected category
                images = os.listdir(os.path.join(dataset_path, category))
                # Randomly select an image from the category
                image_file = random.choice(images)
                # Open and append the selected image
                img = Image.open(os.path.join(dataset_path, category, image_file))
                row_images.append(img)
            
            # Combine the two images horizontally
            combined_image.append(combine_horizontally(row_images[0], row_images[1]))

        # Combine the row images vertically
        new_image = combine_images(combined_image)
        # Save the combined image
        new_image.save(os.path.join(output_path, f"combined_{i}.png"))

    print("Combined images generated successfully!")

def annotation_matrix_to_adjacency_tensor(matrix: pd.DataFrame = None, from_csv = None, transpose = False, sort = None, index=None):
    if from_csv is not None :
        matrix = pd.read_csv(from_csv, index_col=0, header=0)

    if transpose:
        matrix = matrix.T
        
    if sort == "columns":
        matrix = utils.sort_dataframe(matrix, mode=sort, index=index)

    index_row = matrix.index
    index_col = matrix.columns
   
    _, num_cols = matrix.shape
    
    torch_matrix = torch.Tensor(pd.DataFrame.to_numpy(matrix))

    channels = []
    for row in torch_matrix:
        channel = torch.Tensor(num_cols, num_cols).fill_(0)
        indexes = torch.argwhere(row == 1)
        for ind_row in indexes:
            for ind_col in indexes:
                channel[ind_row, ind_col] = 1
        channels.append(channel)

    return torch.stack(channels), list(index_row), list(index_col)


class ImageFolderNoLabel(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = sorted(os.listdir(root))
        
    def get_paths(self):
        return self.images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root, self.images[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image
