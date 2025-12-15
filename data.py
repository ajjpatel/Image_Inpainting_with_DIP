import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from utils import make_mask
from PIL import Image

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp', '.ppm', '.pgm')

def find_images_recursively(root):
    image_paths = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower().endswith(IMG_EXTENSIONS) and not fname.startswith('.'):
                image_paths.append(os.path.join(dirpath, fname))
    return sorted(image_paths)

class InpaintingDataset(Dataset):
    def __init__(self, root, dataset='celeba', image_size=128, mask_size=64, mask_type='mixed', transform=None):
        self.image_size = image_size
        self.mask_size = mask_size
        self.mask_type = mask_type
        if transform:
            transform_list = [transforms.Resize((image_size, image_size)), transform]
        else:
            transform_list = [
                transforms.CenterCrop(min(image_size, image_size)),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        self.transform = transforms.Compose(transform_list)

        if dataset == 'celeba':
            self.samples = find_images_recursively(root)
            if len(self.samples) == 0:
                raise RuntimeError(f"No images found in {root} (searched recursively). Supported extensions: {IMG_EXTENSIONS}")
            print(f"[INFO] Found {len(self.samples)} images for 'celeba' in {root} (recursively, single class)")
            self.use_flat = True
        elif dataset == 'imagenet':
            self.dataset = datasets.ImageFolder(root=os.path.join(root, 'train'), transform=self.transform)
            self.use_flat = False
        else:
            raise ValueError(f"Unsupported dataset {dataset}")

    def __len__(self):
        if hasattr(self, 'use_flat') and self.use_flat:
            return len(self.samples)
        return len(self.dataset)

    def __getitem__(self, idx):
        if hasattr(self, 'use_flat') and self.use_flat:
            img_path = self.samples[idx]
            img = Image.open(img_path) #.convert('RGB')
            img = self.transform(img)
        else:
            img, _ = self.dataset[idx] if isinstance(self.dataset[idx], tuple) else (self.dataset[idx], None)
        mask = make_mask(self.image_size, self.mask_size, self.mask_type)
        masked_img = img.clone()
        masked_img = masked_img * (1 - mask)
        return masked_img, img, mask
