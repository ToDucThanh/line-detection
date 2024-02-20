import json
import os
import pickle
import random

import cv2
import numpy as np
import torch
import tqdm
from albumentations import (
    Compose,
    HueSaturationValue,
    Normalize,
    OneOf,
    RandomBrightnessContrast,
)
from torch.utils.data import Dataset

from .utils import (
    cut_line_by_xmax,
    cut_line_by_xmin,
    gen_junction_and_line_mask,
    gen_SOL_map,
    gen_TP_mask2,
    get_ext_lines,
    swap_line_pt_maybe,
)


def parse_label_file_info(img_dir, label_file):
    infos = []
    contens = json.load(open(label_file, "r"))
    for c in tqdm.tqdm(contens):
        w = c["width"]
        h = c["height"]
        lines = c["lines"]
        fn = c["filename"][:-4] + ".jpg"
        full_fn = img_dir + fn
        assert os.path.exists(full_fn), full_fn

        json_content = {
            "version": "4.5.6",
            "flags": {},
            "shapes": [],
            "imagePath": fn,
            "imageData": None,
            "imageHeight": h,
            "imageWidth": w,
        }
        for l in lines:
            item = {
                "label": "line",
                "points": [
                    [np.clip(np.float64(l[0]), 0, w), np.clip(np.float64(l[1]), 0, h)],
                    [np.clip(np.float64(l[2]), 0, w), np.clip(np.float64(l[3]), 0, h)],
                ],
                "group_id": None,
                "shape_type": "line",
                "flags": {},
            }
            json_content["shapes"].append(item)
        infos.append(json_content)
    return infos


class LineDataset(Dataset):
    def __init__(self, cfg, is_train):
        super(LineDataset, self).__init__()

        self.cfg = cfg
        self.min_len = cfg.decode.len_thresh
        self.is_train = is_train

        self.img_dir = cfg.train.img_dir
        self.label_fn = cfg.train.label_fn

        if not is_train:
            self.img_dir = cfg.val.img_dir
            self.label_fn = cfg.val.label_fn

        self.cache_dir = cfg.train.data_cache_dir
        self.with_cache = cfg.train.with_cache

        print("==> load label ...")
        if self.with_cache:
            ann_cache_fn = (
                self.cache_dir + "/" + os.path.basename(self.label_fn) + ".cache"
            )
            if os.path.exists(ann_cache_fn):
                print("==> load {} from cache dir..".format(ann_cache_fn))
                self.anns = pickle.load(open(ann_cache_fn, "rb"))
            else:
                self.anns = self._load_anns(self.img_dir, self.label_fn)
                print("==> cache to  {}".format(ann_cache_fn))
                pickle.dump(self.anns, open(ann_cache_fn, "wb"))
        else:
            self.anns = self._load_anns(self.img_dir, self.label_fn)

        print("==> valid samples: ", len(self.anns))

        self.input_size = cfg.datasets.input_size
        self.train_aug = self._aug_train()
        self.test_aug = self._aug_test(input_size=self.input_size)

        self.cache_to_mem = cfg.train.cache_to_mem
        self.cache_dict = {}
        if self.with_cache:
            print("===> cache...")
            for ann in tqdm.tqdm(self.anns):
                self.load_label(ann, False)

    def __len__(self):
        return len(self.anns)

    def _aug_train(self):
        aug = Compose(
            [
                OneOf(
                    [
                        HueSaturationValue(
                            hue_shift_limit=10,
                            sat_shift_limit=10,
                            val_shift_limit=10,
                            p=0.5,
                        ),
                        RandomBrightnessContrast(
                            brightness_limit=0.2, contrast_limit=0.2, p=0.5
                        ),
                    ]
                ),
            ],
            p=1.0,
        )
        return aug

    def _aug_test(self, input_size=384):
        aug = Compose(
            [Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))],
            p=1.0,
        )
        return aug

    def _line_len_fn(self, l1):
        len1 = np.sqrt((l1[2] - l1[0]) ** 2 + (l1[3] - l1[1]) ** 2)
        return len1

    def _load_anns(self, img_dir, label_fn):
        infos = parse_label_file_info(img_dir, label_fn)
        anns = []
        for c in infos:
            img_full_fn = os.path.join(img_dir, c["imagePath"])
            if not os.path.exists(img_full_fn):
                print("{} not exist!".format(img_full_fn))
                exit(0)

            lines = []
            for s in c["shapes"]:
                pt = s["points"]
                line = [pt[0][0], pt[0][1], pt[1][0], pt[1][1]]
                line = swap_line_pt_maybe(line)

                if not self.is_train:
                    lines.append(line)
                elif self._line_len_fn(line) > self.min_len:
                    lines.append(line)

            dst_ann = {
                "img_full_fn": img_full_fn,
                "lines": lines,
                "img_w": c["imageWidth"],
                "img_h": c["imageHeight"],
            }
            anns.append(dst_ann)

        return anns

    def _crop_aug(self, img, ann_origin):
        assert img.shape[1] == ann_origin["img_w"]
        assert img.shape[0] == ann_origin["img_h"]
        img_w = ann_origin["img_w"]
        img_h = ann_origin["img_h"]
        lines = ann_origin["lines"]
        xmin = random.randint(1, int(0.1 * img_w))

        xmin_lines = []
        for line in lines:
            flg, line = cut_line_by_xmin(line, xmin)
            line[0] -= xmin
            line[2] -= xmin
            if flg and self._line_len_fn(line) > self.min_len:
                xmin_lines.append(line)
        lines = xmin_lines

        img = img[:, xmin:, :]
        ## xmax
        xmax = img.shape[1] - random.randint(1, int(0.1 * img.shape[1]))
        img = img[:, :xmax, :].copy()
        xmax_lines = []
        for line in lines:
            flg, line = cut_line_by_xmax(line, xmax)
            if flg and self._line_len_fn(line) > self.min_len:
                xmax_lines.append(line)
        lines = xmax_lines

        ann_origin["lines"] = lines
        ann_origin["img_w"] = img.shape[1]
        ann_origin["img_h"] = img.shape[0]

        return img, ann_origin

    def _geo_aug(self, img, ann_origin):
        do_aug = False

        lines = ann_origin["lines"].copy()
        if random.random() < 0.5:
            do_aug = True
            flipped_lines = []
            img = np.fliplr(img)
            for l in lines:
                flipped_lines.append(
                    swap_line_pt_maybe(
                        [
                            ann_origin["img_w"] - l[0],
                            l[1],
                            ann_origin["img_w"] - l[2],
                            l[3],
                        ]
                    )
                )
            ann_origin["lines"] = flipped_lines

        lines = ann_origin["lines"].copy()
        if random.random() < 0.5:
            do_aug = True
            flipped_lines = []
            img = np.flipud(img)
            for l in lines:
                flipped_lines.append(
                    swap_line_pt_maybe(
                        [
                            l[0],
                            ann_origin["img_h"] - l[1],
                            l[2],
                            ann_origin["img_h"] - l[3],
                        ]
                    )
                )
            ann_origin["lines"] = flipped_lines

        lines = ann_origin["lines"].copy()
        if random.random() < 0.5:
            do_aug = True
            r_lines = []
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            for l in lines:
                r_lines.append(
                    swap_line_pt_maybe(
                        [
                            ann_origin["img_h"] - l[1],
                            l[0],
                            ann_origin["img_h"] - l[3],
                            l[2],
                        ]
                    )
                )
            ann_origin["lines"] = r_lines
            ann_origin["img_w"] = img.shape[1]
            ann_origin["img_h"] = img.shape[0]

        if random.random() < 0.5:
            do_aug = True
            img, ann_origin = self._crop_aug(img, ann_origin)

        ann_origin["img_w"] = img.shape[1]
        ann_origin["img_h"] = img.shape[0]

        return do_aug, img, ann_origin

    def load_label(self, ann, do_aug):
        norm_lines = []
        for l in ann["lines"]:
            ll = [
                np.clip(l[0] / ann["img_w"], 0, 1),
                np.clip(l[1] / ann["img_h"], 0, 1),
                np.clip(l[2] / ann["img_w"], 0, 1),
                np.clip(l[3] / ann["img_h"], 0, 1),
            ]
            x0, y0, x1, y1 = 256 * ll[0], 256 * ll[1], 256 * ll[2], 256 * ll[3]
            if x0 == x1 and y0 == y1:
                print("fatal err!")
                print(ann["img_w"], ann["img_h"])
                print(ll)
                print(l)
                print(ann)
                exit(0)

            norm_lines.append(ll)

        ann["norm_lines"] = norm_lines

        label_cache_path = os.path.basename(ann["img_full_fn"])[:-4] + ".npy"
        label_cache_path = self.cache_dir + "/" + label_cache_path

        can_load = self.with_cache and not do_aug

        if (
            can_load
            and self.cache_to_mem
            and label_cache_path in self.cache_dict.keys()
        ):
            label = self.cache_dict[label_cache_path]

        elif can_load and os.path.exists(label_cache_path):
            label = np.load(label_cache_path)
            if self.cache_to_mem:
                self.cache_dict[label_cache_path] = label
        else:
            tp_mask = gen_TP_mask2(
                ann["norm_lines"],
                self.input_size // 2,
                self.input_size // 2,
                with_ext=self.cfg.datasets.with_centermap_extend,
            )
            sol_mask, _ = gen_SOL_map(
                ann["norm_lines"],
                self.input_size // 2,
                self.input_size // 2,
                with_ext=False,
            )

            junction_map, line_map = gen_junction_and_line_mask(
                ann["norm_lines"], self.input_size // 2, self.input_size // 2
            )

            label = np.zeros(
                (2 * 7 + 2, self.input_size // 2, self.input_size // 2),
                dtype=np.float32,
            )
            label[0:7, :, :] = sol_mask
            label[7:14, :, :] = tp_mask
            label[14, :, :] = junction_map[0]
            label[15, :, :] = line_map[0]
            if not do_aug and self.with_cache:
                if self.cache_to_mem:
                    self.cache_dict[label_cache_path] = label
                else:
                    np.save(label_cache_path, label)

        return label

    def __getitem__(self, index):
        ann = self.anns[index].copy()
        img = cv2.imread(ann["img_full_fn"])

        do_aug = False
        if self.is_train and random.random() < 0.5:
            do_aug, img, ann = self._geo_aug(img, ann)

        img = cv2.resize(img, (self.input_size, self.input_size))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = self.load_label(ann, do_aug)
        ext_lines = get_ext_lines(
            ann["norm_lines"], self.input_size // 2, self.input_size // 2
        )

        norm_lines = ann["norm_lines"]
        norm_lines_512_list = []
        for l in norm_lines:
            norm_lines_512_list.append(
                [
                    l[0] * 512,
                    l[1] * 512,
                    l[2] * 512,
                    l[3] * 512,
                ]
            )

        if self.is_train:
            img = self.train_aug(image=img)["image"]
        img_norm = self.test_aug(image=img)["image"]

        norm_lines_512_tensor = torch.from_numpy(
            np.array(norm_lines_512_list, np.float32)
        )
        sol_lines_512_tensor = torch.from_numpy(np.array(ext_lines, np.float32) * 512)

        return (
            img_norm,
            img,
            label,
            norm_lines_512_list,
            norm_lines_512_tensor,
            sol_lines_512_tensor,
            ann["img_full_fn"],
        )


def LineDataset_collate_fn(batch):
    batch_size = len(batch)
    h, w, _ = batch[0][0].shape
    images = np.zeros((batch_size, 3, h, w), dtype=np.float32)
    labels = np.zeros((batch_size, 16, h // 2, w // 2), dtype=np.float32)
    img_fns = []
    img_origin_list = []
    norm_lines_512_all = []
    norm_lines_512_all_tensor_list = []
    sol_lines_512_all_tensor_list = []

    for inx in range(batch_size):
        (
            im,
            img_origin,
            label_mask,
            norm_lines_512,
            norm_lines_512_tensor,
            sol_lines_512,
            img_fn,
        ) = batch[inx]

        images[inx] = im.transpose((2, 0, 1))
        labels[inx] = label_mask
        img_origin_list.append(img_origin)
        img_fns.append(img_fn)
        norm_lines_512_all.append(norm_lines_512)
        norm_lines_512_all_tensor_list.append(norm_lines_512_tensor)
        sol_lines_512_all_tensor_list.append(sol_lines_512)

    images = torch.from_numpy(images)
    labels = torch.from_numpy(labels)

    return {
        "xs": images,
        "ys": labels,
        "img_fns": img_fns,
        "origin_imgs": img_origin_list,
        "gt_lines_512": norm_lines_512_all,
        "gt_lines_tensor_512_list": norm_lines_512_all_tensor_list,
        "sol_lines_512_all_tensor_list": sol_lines_512_all_tensor_list,
    }
