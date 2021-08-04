import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd()))
sys.path.append(os.path.join(os.getcwd(), 'spvnas'))

print(sys.path)

import torchsparse
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize
from torchsparse.utils.collate import sparse_collate, sparse_collate_fn

COLOR_MAP = np.array(['#f59664', '#f5e664', '#963c1e', '#b41e50', '#ff0000',
                      '#1e1eff', '#c828ff', '#5a1e96', '#ff00ff', '#ff96ff',
                      '#4b004b', '#4b00af', '#00c8ff', '#3278ff', '#00af00',
                      '#003c87', '#50f096', '#96f0ff', '#0000ff', '#ffffff'])

LABEL_MAP = np.array([19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 0, 1, 19,
                      19, 19, 2, 19, 19, 3, 19, 4, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 5, 6, 7, 19, 19, 19, 19, 19, 19,
                      19, 8, 19, 19, 19, 9, 19, 19, 19, 10, 11, 12, 13,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 14, 15, 16, 19, 19, 19, 19, 19,
                      19, 19, 17, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19])

# load sample lidar & label
lidar = np.fromfile('./assets/000000.bin', dtype=np.float32)
label = np.fromfile('./assets/000000.label', dtype=np.int32)
lidar = lidar.reshape(-1, 4)
label = LABEL_MAP[label & 0xFFFF]

# filter ignored points
lidar = lidar[label != 19]
label = label[label != 19]

# get rounded coordinates
coords = np.round(lidar[:, :3] / 0.05)
# print(coords)
# print(coords.min(0, keepdims=1))
coords -= coords.min(0, keepdims=1)
feats = lidar
# print(coords)

from itertools import repeat

# # sparse quantization: filter out duplicate points
# if isinstance(feats, (float, int)):
feats = tuple(repeat(feats, 3))

print(coords.shape)
print(np.array(feats).shape)


# print('here')
# print(isinstance(feats, tuple),  len(feats))

indices, inverse = sparse_quantize(coords, feats, return_index=True)

# coords = coords[indices]
# feats = feats[indices]

# # construct the sparse tensor
# inputs = SparseTensor(feats, coords)
# inputs = sparse_collate_tensors([inputs]).cuda()