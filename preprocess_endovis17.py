import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from random import sample

root_path = '/home/zijian/Downloads/endovis2017_instrument_testing' #'/home/zijian/Downloads/endovis2017_instrument_training'
datasets_list = sorted(os.listdir(root_path))
output_root_path = '/home/zijian/Downloads/EndoVis2017'

# dataset_split_ratio = [0.8, 0.2] # list: [train: float, val: float]

for idx, dataset in enumerate(datasets_list): # traverse 4 instrument dataset

    img_root_path = os.path.join(root_path, dataset, 'left_frames')
    imgs_list = sorted(os.listdir(img_root_path))    

    # category_list = sorted(os.listdir(os.path.join(root_path, dataset, 'ground_truth'))) # each dataset has 4 class label

    category_list = ["TypeSegmentation"]

    print(f"-------------{dataset}---------------")
    print(category_list)

    for idx_cls, category in enumerate(category_list): # traverse 4 instrument classes
        gt_root_path = os.path.join(root_path, dataset, 'ground_truth', category)
        gt_list = sorted(os.listdir(gt_root_path))
        breakpoint()

        # valset_num = int(np.round(len(gt_list) * dataset_split_ratio[1]))
        # gt_val = sample(gt_list, valset_num)
        # gt_train = list(set(gt_list) - set(gt_val))

        # new name format: [dataset name]_[dataset number]_[tool class number]_[image number]

        # testing set
        for gt_name in gt_list:
            if imgs_list.count(gt_name) == 0:
                print('Cannot find corresponding image!')
                continue

            idx_gt = gt_name[gt_name.find('.') - 3 : gt_name.find('.')]

            img_path = os.path.join(img_root_path, gt_name)
            img = cv2.imread(img_path)

            gt_path = os.path.join(gt_root_path, gt_name)
            gt = cv2.imread(gt_path)
            breakpoint()
            gt[gt != 0] = 255
            breakpoint()

            img_new_name = 'endovis17' + '_' + str(idx) + '_' + str(idx_cls) + '_' + idx_gt + '.png'
            img_sam_path = os.path.join(output_root_path, 'test', 'images', img_new_name)

            gt_new_name = 'endovis17' + '_' + str(idx) + '_' + str(idx_cls) + '_' + idx_gt + '.png'
            gt_sam_path = os.path.join(output_root_path, 'test', 'masks', gt_new_name)

            cv2.imwrite(img_sam_path, img)
            cv2.imwrite(gt_sam_path, gt)
        # # training set
        # for gt_name in gt_train:
        #     if imgs_list.count(gt_name) == 0:
        #         print('Cannot find corresponding image!')
        #         continue

        #     idx_gt = gt_name[gt_name.find('.') - 3 : gt_name.find('.')]

        #     img_path = os.path.join(img_root_path, gt_name)
        #     img = cv2.imread(img_path)

        #     gt_path = os.path.join(gt_root_path, gt_name)
        #     gt = cv2.imread(gt_path)
        #     gt[gt != 0] = 255

        #     img_new_name = 'endovis17' + '_' + str(idx) + '_' + str(idx_cls) + '_' + idx_gt + '.png'
        #     img_sam_path = os.path.join(output_root_path, 'train', 'images', img_new_name)

        #     gt_new_name = 'endovis17' + '_' + str(idx) + '_' + str(idx_cls) + '_' + idx_gt + '.png'
        #     gt_sam_path = os.path.join(output_root_path, 'train', 'masks', gt_new_name)

        #     cv2.imwrite(img_sam_path, img)
        #     cv2.imwrite(gt_sam_path, gt)
        
        # # validation set
        # for gt_name in gt_val:
        #     if imgs_list.count(gt_name) == 0:
        #         print('Cannot find corresponding image!')
        #         continue

        #     idx_gt = gt_name[gt_name.find('.') - 3 : gt_name.find('.')]

        #     img_path = os.path.join(img_root_path, gt_name)
        #     img = cv2.imread(img_path)

        #     gt_path = os.path.join(gt_root_path, gt_name)
        #     gt = cv2.imread(gt_path)
        #     gt[gt != 0] = 255

        #     img_new_name = 'endovis17' + '_' + str(idx) + '_' + str(idx_cls) + '_' + idx_gt + '.png'
        #     img_sam_path = os.path.join(output_root_path, 'val', 'images', img_new_name)

        #     gt_new_name = 'endovis17' + '_' + str(idx) + '_' + str(idx_cls) + '_' + idx_gt + '.png'
        #     gt_sam_path = os.path.join(output_root_path, 'val', 'masks', gt_new_name)

        #     cv2.imwrite(img_sam_path, img)
        #     cv2.imwrite(gt_sam_path, gt)            

    breakpoint()

