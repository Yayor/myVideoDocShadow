import os

import albumentations as A
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None):
        super(DataLoaderTrain, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'target')))

        self.inp_filenames = [os.path.join(rgb_dir, 'input', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'target', x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        
        self.sizex = len(self.tar_filenames)  # get the size of target

        self.transform = A.Compose([
            A.HorizontalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.ColorJitter(p=0.3),
            A.Affine(p=0.3),
            A.RandomResizedCrop(height=img_options['h'], width=img_options['w']), ],
            additional_targets={
                'target': 'image',
            }
        )

        self.shadow = A.Compose([
            A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_upper=10, shadow_dimension=15, p=1)
        ])

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')

        inp_img = np.array(inp_img)
        tar_img = np.array(tar_img)

        transformed = self.transform(image=inp_img, target=tar_img)

        syn = self.shadow(image=transformed['image'])

        inp_img = F.to_tensor(syn['image'])
        tar_img = F.to_tensor(transformed['target'])

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return inp_img, tar_img, filename


class DataLoaderDSVTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None):
        super(DataLoaderDSVTrain, self).__init__()

        inp_dir = os.path.join(rgb_dir, 'trainA')
        tar_dir = os.path.join(rgb_dir, 'trainB')
        mask_dir = os.path.join(rgb_dir, 'trainC')

        inp_files = sorted([x for x in os.listdir(inp_dir) if is_image_file(x)])
        tar_files = sorted([x for x in os.listdir(tar_dir) if is_image_file(x)])
        mask_files = sorted([x for x in os.listdir(mask_dir) if is_image_file(x)])

        common_files = sorted(set(inp_files) & set(tar_files) & set(mask_files))
        if not common_files:
            raise RuntimeError('No paired DSV files found in trainA/trainB/trainC.')

        self.inp_filenames = [os.path.join(inp_dir, x) for x in common_files]
        self.tar_filenames = [os.path.join(tar_dir, x) for x in common_files]
        self.mask_filenames = [os.path.join(mask_dir, x) for x in common_files]

        self.img_options = img_options
        self.sizex = len(self.inp_filenames)

        self.transform = A.Compose([
            A.PadIfNeeded(min_height=img_options['h'], min_width=img_options['w'], border_mode=0, value=255, mask_value=0),
            A.RandomCrop(height=img_options['h'], width=img_options['w']),
        ], additional_targets={
            'target': 'image',
        })

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]
        mask_path = self.mask_filenames[index_]

        inp_img = np.array(Image.open(inp_path).convert('RGB'))
        tar_img = np.array(Image.open(tar_path).convert('RGB'))
        mask_img = np.array(Image.open(mask_path).convert('L'))

        transformed = self.transform(image=inp_img, target=tar_img, mask=mask_img)

        inp_img = F.to_tensor(transformed['image'])
        tar_img = F.to_tensor(transformed['target'])
        mask_img = F.to_tensor(transformed['mask'])

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return inp_img, tar_img, mask_img, filename


class DataLoaderDSVVal(Dataset):
    def __init__(self, rgb_dir, img_options=None):
        super(DataLoaderDSVVal, self).__init__()

        inp_dir = os.path.join(rgb_dir, 'testA')
        tar_dir = os.path.join(rgb_dir, 'testB')

        inp_files = sorted([x for x in os.listdir(inp_dir) if is_image_file(x)])
        tar_files = sorted([x for x in os.listdir(tar_dir) if is_image_file(x)])
        common_files = sorted(set(inp_files) & set(tar_files))
        if not common_files:
            raise RuntimeError('No paired DSV files found in testA/testB.')

        self.inp_filenames = [os.path.join(inp_dir, x) for x in common_files]
        self.tar_filenames = [os.path.join(tar_dir, x) for x in common_files]

        self.img_options = img_options
        self.sizex = len(self.inp_filenames)

        self.transform = A.Compose([
            A.Resize(height=img_options['h'], width=img_options['w']),
        ], additional_targets={
            'target': 'image',
        })

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')

        if not self.img_options['ori']:
            inp_img = np.array(inp_img)
            tar_img = np.array(tar_img)

            transformed = self.transform(image=inp_img, target=tar_img)

            inp_img = transformed['image']
            tar_img = transformed['target']

        inp_img = F.to_tensor(inp_img)
        tar_img = F.to_tensor(tar_img)

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return inp_img, tar_img, filename


class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, img_options=None):
        super(DataLoaderVal, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'target')))

        self.inp_filenames = [os.path.join(rgb_dir, 'input', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'target', x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target

        self.transform = A.Compose([
            A.Resize(height=img_options['h'], width=img_options['w']), ],
            additional_targets={
                'target': 'image',
            }
        )

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')

        if not self.img_options['ori']:
            inp_img = np.array(inp_img)
            tar_img = np.array(tar_img)

            transformed = self.transform(image=inp_img, target=tar_img)

            inp_img = transformed['image']
            tar_img = transformed['target']

        inp_img = F.to_tensor(inp_img)
        tar_img = F.to_tensor(tar_img)

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return inp_img, tar_img, filename
