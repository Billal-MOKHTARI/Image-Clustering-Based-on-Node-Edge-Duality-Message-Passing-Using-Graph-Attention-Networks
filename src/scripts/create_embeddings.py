import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from data_loader import create_combined_images

parser = argparse.ArgumentParser(prog='create_embeddings', description='Create imgae embeddings and save them in neptune.')

# parser.add_argument('--model', type=str, help='Path to the dataset directory', required=True)
# parser.add_argument('--destination', type=str, help='Path to the output directory', required=True)
# parser.add_argument('--rows', type=int, help='Number of rows in the combined image', required=False, default=2)
# parser.add_argument('--cols', type=int, help='Number of columns in the combined image', required=False, default=2)
# parser.add_argument('--num_images', type=int, help='Number of combined images to generate', required=False, default=1000)

# args = parser.parse_args()
# create_combined_images(args.source, args.destination, int(args.rows), int(args.cols), int(args.num_images))