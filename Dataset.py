from torch.utils.data import Dataset
import cv2
import os, sys
from utils import get_transforms, PROJECT_ROOT, denorm, device
import matplotlib.pyplot as plt
import random
import torch

class UnlabelledTrainingDataset(torch.utils.data.Dataset):
  def __init__(self, image_dirs, img_size:int=512):
    """
    Dataset for unlabelled training images.
    Args:
        image_dirs (list): List of directories containing training images
        img_size (int): Size to which images will be resized (default: 512)

    Returns:
        torch.utils.data.Dataset: The unlabelled training dataset    
    """
    super().__init__()

    print('Intializing UnlabelledTrainingDataset...')
    print('Image directories: {}'.format((', '.join(image_dirs))))
    print('Checking if directories exist...')

    self.image_dirs = image_dirs

    for dir_path in (self.image_dirs):
      if len(self.image_dirs) == 0:
        raise ValueError("No image directories provided.")
      else:
        i = 0
        dir_paths = []
        for dir_path in self.image_dirs:
            dir_paths.append(dir_path)
            if not os.path.exists(dir_path):
                dir_path = os.path.join(PROJECT_ROOT, dir_path)
                raise FileNotFoundError("Directorie(s) not found: {}".format(', '.join(dir_path)))
            else:
                i += 1
    print("{} Directorie(s) found: {}".format(i, ', '.join(dir_paths)))
    
    print('Preparing image paths...')

    self.image_size = 512

    self.image_paths = []
    self.transforms = get_transforms('train')

    self.clahe = cv2.createCLAHE(2.0,(8,8))

    for dirs in self.image_dirs:
      for root, dirs, files in os.walk(dirs):
        for f in files:
          self.image_paths.append(os.path.join(root, f))

  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self, idx):
    img = cv2.imread(self.image_paths[idx])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    green = img[:,:,1]
    applied_clahe = self.clahe.apply(green)
    combined = cv2.GaussianBlur(applied_clahe,(3,3),0.5)

    transformed = self.transforms(image=combined)
    img = transformed['image']

    return img

class VesselSegmentationDataset(torch.utils.data.Dataset):
  """
    Dataset for vessel segmentation
  """

  def __init__(self, test_dir, train_dir, split):
    super().__init__()

    if len(test_dir) < 2 or len(train_dir) < 2:
        raise ValueError("Please provide valid directories for test and train data. Each directory should contain subdirectories 'images' and 'masks'.")
    
    if isinstance(test_dir, tuple(str)) and isinstance(train_dir, tuple(str)):
        self.test_dir = os.path.join(PROJECT_ROOT, test_dir[0])
        self.train_dir = os.path.join(PROJECT_ROOT, train_dir[0])
        self.mask_test_dir = os.path.join(PROJECT_ROOT, test_dir[1])
        self.mask_train_dir = os.path.join(PROJECT_ROOT, train_dir[1])
    else:
       raise TypeError("Please provide valid directories for test and train data as tuples of strings.")

    if split == 'train':
      self.image_dir = self.train_dir
      self.mask_dir = self.mask_train_dir
    elif split == 'test':
      self.image_dir = self.test_dir
      self.mask_dir = self.mask_test_dir

    print('Preparing image and mask paths for {} data...'.format(split))
    print('Image directory: {}'.format(self.image_dir))
    print('Mask directory: {}'.format(self.mask_dir))


    self.image_paths = sorted(os.listdir(self.image_dir))
    self.mask_paths = sorted(os.listdir(self.mask_dir))

    self.transform = get_transforms('train' if split == 'train' else 'test')

  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self, idx):
    img_path = os.path.join(self.image_dir, self.image_paths[idx])
    mask_path = os.path.join(self.mask_dir, self.mask_paths[idx])

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    green = image[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gaussian_blur = cv2.GaussianBlur(green, (5,5), 0.5)
    combined = clahe.apply(gaussian_blur)

    mask = cv2.imread(mask_path, 0)

    transformed = self.transform(image=combined, mask=mask)
    image = transformed['image']
    mask = transformed['mask']

    if image.ndim == 2:
      image = image.unsqueeze(0)
    if mask.ndim == 2:
      mask = mask.unsqueeze(0)

    mask = (mask > 127).float()


    return image, mask

if __name__ == "__main__":
    dataset = UnlabelledTrainingDataset(image_dirs=['images'])
    print("Dataset length: ", len(dataset))
    
    rand_idx = random.randint(0, len(dataset)-1)
    sample = dataset[rand_idx]
    print("Sample shape: ", sample.shape)
    
    denorm_img = denorm(sample)
    plt.imshow(denorm_img, cmap='gray')
    plt.axis('off')
    plt.show()

    sdataset = VesselSegmentationDataset(test_dir=('test/images', 'test/masks'), train_dir=('train/images', 'train/masks'), split='train')
    print("Segmentation Dataset length: ", len(sdataset))
    s_rand_idx = random.randint(0, len(sdataset)-1)
    s_image, s_mask = sdataset[s_rand_idx]
    print("Segmentation Sample image shape: ", s_image.shape)
    print("Segmentation Sample mask shape: ", s_mask.shape)

    s_denorm_img = denorm(s_image)
    plt.subplot(1, 2, 1)
    plt.imshow(s_denorm_img, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(s_mask.squeeze(0), cmap='gray')
    plt.axis('off')
    plt.show()
    