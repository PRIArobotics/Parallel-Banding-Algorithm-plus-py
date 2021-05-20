import os, sys

os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin")
sys.path.append(r'C:\Users\PRIA\source\repos\Parallel-Banding-Algorithm-plus-py\x64\Debug')

import pba2d
import pba3d
import numpy as np
import torch

points = np.array([
	[0, 0],
	[2047, 2047],
])

# empty input array
arr = np.full([2048, 2048, 2], pba2d.MARKER, dtype=np.short)
# put the points at their positions
arr[points[:, 0], points[:, 1]] = points

cuda0 = torch.device('cuda:0')
tensor = torch.tensor([0, 0], dtype=torch.short, device=cuda0)

print(arr)
print(tensor)

# Compute 2D Voronoi diagram
# Input: a 2D texture. Each pixel is represented as two "short" integer. 
#    For each site at (x, y), the pixel at coordinate (x, y) should contain 
#    the pair (x, y). Pixels that are not sites should contain the pair (MARKER, MARKER)
# See original paper for the effect of the three parameters: m1, m2, m3
# Parameters must divide textureSize
pba2d.voronoi(arr, 32, 32, 2)
print(arr)
