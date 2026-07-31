# datasets.py  (RandAug removed – augmentation now injected from training script)
"""Datasets that leave heavy augmentation (RandAug/MixUp/CutMix) to the trainer.
Only **basic transform** (Resize→ToTensor) is applied here; any extra transforms
should be passed via the `transform=` argument when instantiating the dataset.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Sequence
from typing import Sequence, Dict, Optional
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
import numpy as np

# _TO_TENSOR = transforms.ToTensor()
# _cutmix_tf = transforms.RandomApply(
#     [transforms.RandAugment(num_ops=1, magnitude=6)], p=0.0)  # placeholder
# _allaug_tf = transforms.Compose([
#     transforms.RandAugment(num_ops=2, magnitude=7),
#     transforms.ColorJitter(0.2, 0.2, 0.2)
# ])
# -------------------------------- Label configs ---------------------------------
NUM_CLASSES_TONGUE: Dict[str, int] = {
    '舌质_神': 2, '舌质_色': 5, '舌质_形': 7, '舌质_态': 6,
    '舌苔_苔色': 6, '舌苔_苔质': 8,
}
NUM_CLASSES_FACE: Dict[str, int] = {
    '望眼_目色': 4, '望眼_目态': 5, '望眼_目形': 5, '望眼_瞳孔': 3,
    '望口_口形': 5, '望口_口态': 7,
    '望唇_唇色': 7, '望唇_唇形': 7,
    '望鼻_鼻色': 5, '望鼻_鼻形': 5,
    '面色_面色': 21, '面色_皮肤光泽': 2, '面色_面形': 7,
}
TASK_COLS_TONGUE: List[str] = list(NUM_CLASSES_TONGUE.keys())
TASK_COLS_FACE:   List[str] = list(NUM_CLASSES_FACE.keys())

ALL_LABELS = {
    '舌质_神'   : ['枯舌', '荣舌'],
    '舌质_色'   : ['淡红', '红', '淡白', '绛红', '青紫'],
    '舌质_形'   : ['老', '嫩', '胖', '瘦', '齿痕', '点刺', '裂纹'],
    '舌质_态'   : ['痿软', '强硬', '歪斜', '颤动', '吐弄', '短缩'],
    '舌苔_苔色' : ['灰黑', '白', '黄', '无', '滑', '少'],
    '舌苔_苔质' : ['燥', '薄', '厚', '润', '腻', '腐', '剥落', '偏全'],
    '望眼_目色'       : ['目胞色黑晦暗', '白睛发黄', '两眦淡白', '目赤肿痛'],
    '望眼_目态'       : ['目睛凝视', '胞睑下垂', '睡眠露睛', '目睛上视', '斜视'],
    '望眼_目形'       : ['眼球凹陷', '眼球突出', '胞睑红肿', '目胞浮肿', '无异常'],
    '望眼_瞳孔'       : ['瞳孔缩小', '瞳孔等大', '瞳孔散大'],
    '望口_口形'       : ['口角无异常', '口歪不收', '口疮', '口糜', '鹅口疮'],
    '望口_口态'       : ['口张', '口噤', '口撮', '口', '口振', '口动', '无异常'],
    '望唇_唇色'       : ['淡白', '深红', '赤红', '樱桃红', '青紫', '青黑', '红润'],
    '望唇_唇形'       : ['无异常', '唇干而裂', '嘴唇糜烂', '唇内溃烂', '唇边生疮', '唇角生疔', '口唇翻卷不能覆齿'],
    '望鼻_鼻色'       : ['色白', '色赤', '微黄', '灰暗枯槁', '红黄隐隐 明润含蓄'],
    '望鼻_鼻形'       : ['鼻头生疖', '生粉刺', '鼻翼扇动', '鼻柱溃陷', '无异常'],
    '面色_面色'  : ['青黄','红赤满面通红','午后潮红','久病苍白却时颧赤泛红如妆','淡白无华','晄白','苍白','淡青','青黑','青灰','青黄','黧黑晦暗','紫暗黧黑','黑而干焦','眼眶发黑','萎黄','黄胖','阳黄','阴黄','面黄','红黄隐隐明润含蓄'],
    '面色_皮肤光泽': ['荣润','枯槁'],
    '面色_面形'   : ['无异常','面肿','腮肿','面削颧耸','口眼斜','惊恐貌','苦笑貌'],
}
# -------------------------------- util ---------------------------------
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
def norm(x): return (x - IMAGENET_MEAN) / IMAGENET_STD

def str_to_multihot(idx_str: str, n_cls: int) -> torch.Tensor:
    y = torch.zeros(n_cls, dtype=torch.float32)
    if idx_str:
        y[list(map(int, idx_str.split()))] = 1.
    return y

class _BaseDataset(Dataset):
    """
    Generic image‐classification dataset with:
      - CSV of image names  label columns
      - on‐the‐fly Resize → (optional) RandAugment → PIL→Tensor → extra transforms
      - five‐fold splits handled externally
    """

    def __init__(
        self,
        csv_path: str,
        img_root: str,
        task_cols: Sequence[str],
        num_classes: Dict[str, int],
        *,
        img_size: int = 224,
        ra_prob: float = 0.0,
        ra_n: int = 2,
        ra_m: int = 9,
        extra_tf: Optional[transforms.Compose] = None
    ):
        super().__init__()
        # --- load CSV & build filename list
        self.df = pd.read_csv(csv_path, dtype=str).fillna('')
        self.fnames = [
            re.sub(r'\s', '', row['image'])
            for _, row in self.df.iterrows()
        ]

        # --- build image‐stem → full‐path map
        self.root = Path(img_root).expanduser().resolve()
        self.id2p: Dict[str, Path] = {}
        for p in self.root.rglob('*'):
            if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'}:
                self.id2p[p.stem] = p
                self.id2p[p.name] = p

        # --- labels configuration
        self.task_cols = list(task_cols)
        self.nc = num_classes

        # --- transforms: Resize → optional RandAug → PIL→Tensor → extra
        def _pil2tensor(pic: Image.Image) -> torch.Tensor:
            # pic → H×W×C uint8 np.array → torch.uint8 tensor → C×H×W float32 [0..1]
            arr = np.array(pic, dtype=np.uint8)
            t = torch.tensor(arr, dtype=torch.uint8)
            return t.permute(2, 0, 1).float().div(255.0)

        tf_list = [transforms.Resize((img_size, img_size))]
        if ra_prob > 0:
            tf_list.append(
                transforms.RandomApply(
                    [transforms.RandAugment(num_ops=ra_n, magnitude=ra_m)],
                    p=ra_prob
                )
            )
        tf_list.append(transforms.Lambda(_pil2tensor))
        if extra_tf is not None:
            tf_list.append(extra_tf)
        self.tf = transforms.Compose(tf_list)

    def __len__(self) -> int:
        return len(self.df)

    def _encode_row(self, row: pd.Series) -> Dict[str, torch.Tensor]:
        enc: Dict[str, torch.Tensor] = {}
        for t in self.task_cols:
            idx_str = row[t].strip()
            enc[t] = torch.tensor(int(idx_str), dtype=torch.long)
        return enc

    def _load(self, idx: int) -> torch.Tensor:
        name = re.sub(r'\s', '', self.df.iloc[idx]['image'])
        p = self.id2p.get(name)
        if p is None:
            # try adding common extensions
            for ext in ('.png', '.jpg', '.jpeg', '.bmp'):
                cand = self.root / f'{name}{ext}'
                if cand.exists():
                    p = cand
                    break
        if p is None:
            raise FileNotFoundError(f"Image '{name}' not found under {self.root}")
        return self.tf(Image.open(p).convert('RGB'))

    def __getitem__(self, idx: int):
        img = self._load(idx)
        row = self.df.iloc[idx]
        labels = self._encode_row(row)
        return img, labels, self.fnames[idx]

class TongueDataset(_BaseDataset):
    def __init__(self, csv_path:str, img_root:str, **kwargs):
        super().__init__(csv_path, img_root, TASK_COLS_TONGUE, NUM_CLASSES_TONGUE, **kwargs)
    
def pil_to_tensor(img):
    # img: PIL.Image (RGB)
    # Build a fresh float32 tensor via torch.tensor() to bypass from_numpy
    arr = np.array(img, dtype=np.float32) / 255.0      # H x W x 3
    arr = arr.transpose(2, 0, 1)                       # 3 x H x W
    return torch.tensor(arr, dtype=torch.float32)      # copy -> no shared memory

class MultiViewTongueDataset(_BaseDataset):
    """
    Returns *three* views for every image so the trainer can feed only
    the heads that benefit from each augmentation.
        - img['base']    : resize→tensor (no heavy aug)               (used by 舌质_神 etc.)
        - img['cutmix']  : same tensor (CutMix applied later)         (舌质_形  舌质_态  舌苔_苔色)
        - img['allaug']  : RandAug  jitter                           (舌苔_苔质)

    MixUp is batch‑level, so it's applied inside the training step only
    to the 舌质_色 head; we therefore don't need a dedicated “mixup” view.
    """
    def __init__(self, csv_path:str, img_root:str, is_train: bool, img_size=224,**kwargs):
        super().__init__(csv_path, img_root, TASK_COLS_TONGUE, NUM_CLASSES_TONGUE, extra_tf=None, **kwargs)

        self.img_size = img_size
        # self._basic_tf = transforms.Resize((img_size, img_size))
        self.is_train = is_train
        if is_train:
            self.basic_tf = transforms.Compose([
                transforms.RandomResizedCrop(256, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(0.5),
            ])
        else:
            self.basic_tf = transforms.Resize((256, 256))
        # all-aug for 舌苔_苔质 (train only); at val we just reuse base
        self.allaug_tf = transforms.Compose([
            transforms.RandAugment(num_ops=2, magnitude=7),
            transforms.ColorJitter(0.2, 0.2, 0.2),  # no hue (your env issue)
        ])  

    def _load_views(self, idx:int):
        name = re.sub(r'\s+', '', self.df.iloc[idx]['image'])
        p = self.id2p.get(name)
        if p is None:
            for ext in ('.png', '.jpg', '.jpeg', '.bmp'):
                cand = self.root / f"{name}{ext}"
                if cand.exists():
                    p = cand
                    break
        if p is None:
            raise FileNotFoundError(name)

        img = Image.open(p).convert('RGB')

        # base view
        base_img = self.basic_tf(img)
        base = pil_to_tensor(base_img)

        # cutmix view: just clone base (CutMix applied in train loop)
        cut = base.clone()

        # all-aug view: apply augment then resize then to tensor
        if self.is_train:
            aug_img = self.basic_tf(self.allaug_tf(img))
            allaug = pil_to_tensor(aug_img)
        else:
            allaug = base.clone()
        

        # aug_img = _allaug_tf(img)
        # aug_img = self._basic_tf(aug_img)
        # aug = pil_to_tensor(aug_img)
        base = norm(base); cut = norm(cut); allaug = norm(allaug)
        return {"base": base, "cutmix": cut, "allaug": allaug}
    def __getitem__(self, idx:int):
        views = self._load_views(idx)
        labels = self._encode_row(self.df.iloc[idx])
        return views, labels, self.fnames[idx]

# -------------------------------------------------------------------------------
# DataLoader helper
# -------------------------------------------------------------------------------

def collate_multiview(batch):
    # batch = List[(views_dict, label_dict, fname)]
    view_keys = batch[0][0].keys()        # e.g. ("base","cutmix","allaug")
    collated = {k: torch.stack([b[0][k] for b in batch]) for k in view_keys}
    labels   = {t: torch.tensor([b[1][t] for b in batch], dtype=torch.long)
                for t in batch[0][1]}
    fnames   = [b[2] for b in batch]
    return collated, labels, fnames        


class FaceDataset(_BaseDataset):
    def __init__(self, csv_path:str, img_root:str, **kwargs):
        super().__init__(csv_path, img_root, TASK_COLS_FACE, NUM_CLASSES_FACE, **kwargs)

from torch.utils.data import DataLoader

def collate_multitask(batch):
    imgs, lab_dicts, fnames = zip(*batch)
    imgs = torch.stack(imgs)
    out  = {t: torch.tensor([d[t] for d in lab_dicts], dtype=torch.long)
            for t in lab_dicts[0]}
    return imgs, out, list(fnames)


