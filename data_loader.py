import torch
import numpy as np
import torch.nn.functional as F
import os

from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from torchvision import transforms


def dense_diffusion_loader(dataset, inst_gt, idx, layout_img_root, diffusion_pipeline, tokenizer, sp_sz):
    bsz = 1
    device_ = diffusion_pipeline.module.device

    # prep for cls_gt & mask_gt
    cls_gt = inst_gt[idx]["cls_gt"]
    mask_gt = inst_gt[idx]["mask_gt"]

    layout_img_path = layout_img_root + str(idx) + '.png'
    prompts = [dataset[idx]['textual_condition']] + dataset[idx]['segment_descriptions']
    # shape: (1, 77, 768)
    uncond_embed = diffusion_pipeline.module.get_learned_conditioning([""])
    # shape: (4 or 5, 77, 768)
    cond_embed = diffusion_pipeline.module.get_learned_conditioning(prompts)
    # shape: (512, 512, 3) -> (512, 512, 3)
    layout_img_ = np.asarray(Image.open(layout_img_path).resize([sp_sz * 8, sp_sz * 8]))[:, :, :3]
    unique, counts = np.unique(np.reshape(layout_img_, (-1, 3)), axis=0, return_counts=True)
    sorted_idx = np.argsort(-counts)

    # one class one mask
    layouts_ = []
    for i in range(len(prompts) - 1):
        if (unique[sorted_idx[i]] == [0, 0, 0]).all() or (unique[sorted_idx[i]] == [255, 255, 255]).all():
            # background
            layouts_ = [((layout_img_ == unique[sorted_idx[i]]).sum(-1) == 3).astype(np.uint8)] + layouts_
        else:
            # foreground
            layouts_.append(((layout_img_ == unique[sorted_idx[i]]).sum(-1) == 3).astype(np.uint8))

    layouts = [torch.FloatTensor(l).unsqueeze(0).unsqueeze(0).cuda() for l in layouts_]
    layouts = F.interpolate(torch.cat(layouts), (sp_sz, sp_sz), mode='nearest')
    # print(layouts.shape)
    # print('\n'.join(prompts))
    # Image.fromarray(np.concatenate([255 * _.squeeze().cpu().numpy() for _ in layouts], 1).astype(np.uint8))

    # prep for sreg (self-attention map)
    # scale object size
    reg_sizes = {}
    sreg_maps = {}
    for r in range(4):
        # 64, 32, 16, 8, ..., 64, 32, 16, 8
        res = int(sp_sz / np.power(2, r))
        layouts_s = F.interpolate(layouts, (res, res), mode='nearest')
        layouts_s = (layouts_s.view(layouts_s.size(0), 1, -1) * layouts_s.view(layouts_s.size(0), -1, 1)).sum(
            0).unsqueeze(0).repeat(bsz, 1, 1)
        # 4096, 1024, 256, 64, ..., 4096, 1024, 256, 64
        # len(reg_sizes) = len(sreg_maps) = 4
        reg_sizes[np.power(res, 2)] = 1 - 1. * layouts_s.sum(-1, keepdim=True) / (np.power(res, 2))
        sreg_maps[np.power(res, 2)] = layouts_s

    # prep for creg (cross-attention map)
    pww_maps = torch.zeros(1, 77, sp_sz, sp_sz).to(device_)
    text_input = tokenizer(prompts, padding="max_length", return_length=True, return_overflowing_tokens=False,
                           max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    # prompts, e.g.,
    # 'A bicycle replica with a clock as the front wheel.', '', 'A bicycle replica', 'a clock as the front wheel'
    # length [13, 2, 5, 8]
    for i in range(1, len(prompts)):
        wlen = text_input['length'][i] - 2
        # only meaningful word
        widx = text_input['input_ids'][i][1:1 + wlen]
        for j in range(77):
            if (text_input['input_ids'][0][j:j + wlen] == widx).sum() == wlen:
                pww_maps[:, j:j + wlen, :, :] = layouts[i - 1:i]
                cond_embed[0][j:j + wlen] = cond_embed[i][1:1 + wlen]
                # print(prompts[i], i, '-th segment is handled.')
                break
    creg_maps = {}
    for r in range(4):
        res = int(sp_sz / np.power(2, r))
        layout_c = F.interpolate(pww_maps, (res, res), mode='nearest').view(1, 77, -1).permute(0, 2, 1).repeat(bsz,
                                                                                                               1, 1)
        creg_maps[np.power(res, 2)] = layout_c

    # prep for text_emb
    text_cond = torch.cat([uncond_embed, cond_embed[:1].repeat(bsz, 1, 1)])

    # cls_gt
    # mask_gt
    # reg_size
    # sreg_maps
    # creg_maps
    # text_cond
    return cls_gt, mask_gt, reg_sizes, sreg_maps, creg_maps, text_cond, dataset[idx]['textual_condition']


class CocoCaptionDataset(Dataset):
    def __init__(self,
                 base_dir="dataset/coco/",
                 split="val",
                 year="2017",
                 image_yres=512,
                 image_xres=512
                 ):
        super().__init__()
        self.ann_file_path = os.path.join(base_dir, "annotations/captions_{}{}.json".format(split, year))
        self.image_dir = os.path.join(base_dir, "{}{}".format(split, year))
        self.coco_annotations = COCO(self.ann_file_path)
        self.img_ids = list(self.coco_annotations.imgs.keys())

        self.image_trans = transforms.Compose([
            transforms.Resize((image_yres, image_xres)),
            # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def _make_image_caption_pair(self, idx):
        coco_annotations = self.coco_annotations
        image_idx = self.img_ids[idx]
        image_metadata = coco_annotations.loadImgs(image_idx)[0]
        image_path = image_metadata["file_name"]
        image = Image.open(os.path.join(self.image_dir, image_path)).convert("RGB")
        # 5 caption per image
        # len(caption) = 5
        # image_id; id (sentence ?), caption
        caption = coco_annotations.loadAnns(coco_annotations.getAnnIds(image_idx))
        diverse_captions = []
        for k in range(len(caption)):
            diverse_captions.append(caption[k])
        return image, diverse_captions, image_idx

    def __getitem__(self, idx):
        image, caption, image_idx = self._make_image_caption_pair(idx)
        image = self._image_preprocessing(image)
        return image, caption, image_idx

    def _image_preprocessing(self, image):
        image = self.image_trans(image)
        return image

    def __len__(self):
        return len(self.img_ids)
