import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
rootPath = os.path.split(rootPath)[0]
# print(rootPath)
sys.path.append(rootPath)
# print(os.listdir(os.path.join('data/floodnet', 'test_floodnet', 'image_p1024s1024')))

import numpy as np
from PIL import Image
from mypath import Path
from torch.utils import data
from dataloaders import custom_transforms as tr


def twoTrainSeg(args, root=Path.db_root_dir('floodnet')):
    images_base = os.path.join(root, 'train_floodnet', 'image_p1024s1024')
    train_files = [os.path.join(looproot, filename) for looproot, _, filenames in os.walk(images_base)
                   for filename in filenames if filename.endswith('.jpg')]
    number_images = len(train_files)
    permuted_indices_ls = np.random.permutation(number_images)
    indices_1 = permuted_indices_ls[: int(0.5 * number_images) + 1]
    indices_2 = permuted_indices_ls[int(0.5 * number_images):]
    # if len(indices_1) % 2 != 0 or len(indices_2) % 2 != 0:
    #     raise Exception('indices lists need to be even numbers for batch norm')
    return FloodnetSegmentation(args, split='train', indices_for_split=indices_1),  FloodnetSegmentation(args, split='train', indices_for_split=indices_2)


class FloodnetSegmentation(data.Dataset):
    NUM_CLASSES = 10

    CLASSES = [
        'background', 'building_flooded', 'building_no_flooded', 'road_flooded', 'road_non_flooded', 'water', 'tree', 'vehicle', 'pool', 'grass'
    ]

    def __init__(self, args, root=Path.db_root_dir('floodnet'), split="train", indices_for_split=None):
        self.root = root
        self.split = split
        self.args = args
        self.files = {}
        self.mean = (123.675, 116.28, 103.53)
        self.std = (58.395, 57.12, 57.375)
        self.crop = self.args.crop_size
        if split.startswith('re'):
            self.images_base = os.path.join(self.root, self.split[2:]+'_floodnet', 'image_p1024s1024')
            self.annotations_base = os.path.join(self.root, self.split[2:]+'_floodnet', 'label_p1024s1024')
        else:
            self.images_base = os.path.join(self.root, self.split+'_floodnet', 'image_p1024s1024')
            self.annotations_base = os.path.join(self.root, self.split+'_floodnet', 'label_p1024s1024')

        self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix='.jpg')
        # print(len(self.files[split]))
        # sys.exit()

        #对得到的train数据索引重排 defeaut = None
        if indices_for_split is not None:
            self.files[split] = np.array(self.files[split])[indices_for_split].tolist()

        self.void_classes = []
        self.valid_classes = [0,1,2,3,4,5,6,7,8,9]
        self.class_names = ['background', 'building_flooded', 'building_no_flooded', 'road_flooded', 'road_non_flooded', 'water', 'tree', 'vehicle', 'pool', 'grass']

        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))
        # print(self.class_map)
        # sys.exit()

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))
        self.transform = self.get_transform()

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):

        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(self.annotations_base, img_path.split(os.sep)[-1].replace('jpg', 'png'))

        _img = Image.open(img_path).convert('RGB')
        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
        _tmp = self.encode_segmap(_tmp)
        _target = Image.fromarray(_tmp)

        sample = {'image': _img, 'label': _target}
        return self.transform(sample)

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]

    def get_transform(self):
        if self.split == 'train':
            return tr.transform_tr(self.args, self.mean, self.std)
        elif self.split == 'test':
            return tr.transform_val(self.args, self.mean, self.std)
        elif self.split == 'retrain':
            return tr.transform_retr_floodnet(self.args, self.mean, self.std)
        elif self.split == 'retest':
            return tr.transform_reval(self.args, self.mean, self.std)


if __name__ == '__main__':
    from dataloaders.dataloader_utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.resize = 512
    args.crop_size = 321

    cityscapes_train = FloodnetSegmentation(args, split='test')

    dataloader = DataLoader(cityscapes_train, batch_size=8, shuffle=True, num_workers=1)

    for ii, sample in enumerate(dataloader):
        print(sample["image"].size())
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()

            tmp = np.array(gt[jj]).astype(np.uint8)
            print(tmp.shape)

            segmap = decode_segmap(tmp, dataset='floodnet')
            print(segmap.shape)

            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (58.395, 57.12, 57.375)
            img_tmp += (123.675, 116.28, 103.53)
            img_tmp *= 255
            print(np.max(img_tmp), np.min(img_tmp))
            img_tmp = img_tmp.astype(np.uint8)
            
            plt.figure(ii * jj + jj, figsize=(10,8))
            plt.title('display')
            plt.subplot(121)
            plt.imshow(img_tmp)
            plt.subplot(122)
            plt.imshow(segmap)

            if jj == 4:
                break
        if ii == 0:
            break

    plt.show(block=True)


