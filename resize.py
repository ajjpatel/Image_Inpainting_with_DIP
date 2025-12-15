import argparse
import os
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Resize all images in a folder to 256x256")
    parser.add_argument('input_dir', type=str, help='Path to input folder containing images')
    parser.add_argument('output_dir', type=str, help='Path to output folder for resized images')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    image_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    
    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(image_extensions)]

    for img_file in image_files:
        input_path = os.path.join(args.input_dir, img_file)
        img = Image.open(input_path).convert('RGB')
        img_resized = img.resize((256, 256), Image.Resampling.LANCZOS)
        output_path = os.path.join(args.output_dir, img_file)
        img_resized.save(output_path)

    print(f"Completed! Resized images saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
