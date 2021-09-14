import os, sys

os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin")
sys.path.append(r'C:\Users\PRIA\source\repos\Parallel-Banding-Algorithm-plus-py\x64\Debug')

import pba2d
import pba3d
import numpy as np

def pytorch():
	# conditional import because it takes long
	import torch

	cuda0 = torch.device('cuda:0')
	tensor = torch.tensor([0, 0], dtype=torch.short, device=cuda0)

	print(tensor)

def voronoi2d():
	points = np.array([
		[0, 1],
		[2047, 2047],
	])

	# empty input array
	arr = np.full([2048, 2048, 2], pba2d.MARKER, dtype=np.short)
	# put the points at their positions
	arr[points[:, 1], points[:, 0]] = points


	print(arr)

	# Compute 2D Voronoi diagram
	# Input: a 2D texture. Each pixel is represented as two "short" integer. 
	#    For each site at (x, y), the pixel at coordinate (x, y) should contain 
	#    the pair (x, y). Pixels that are not sites should contain the pair (MARKER, MARKER)
	# See original paper for the effect of the three parameters: m1, m2, m3
	# Parameters must divide textureSize
	pba2d.voronoi(arr, 32, 32, 2)
	print(arr)


def voronoi3d(omega):
	""" omega is np 3d-array """
	def encoded(p):
		return np.left_shift(p[..., 0], 20) | np.left_shift(p[..., 1], 10) | p[..., 2]

	def decoded(p):
		return np.stack([np.right_shift(p, 20) & 1023, np.right_shift(p, 10) & 1023, p & 1023], axis=-1)

	"""points = np.array([
		[0, 1, 2],
		[511, 511, 511],
	])"""
	points = np.transpose(np.where(omega))

	# empty input array
	arr = np.full(omega.shape, pba3d.MARKER, dtype=int)
	# put the points at their positions
	arr[points[:, 2], points[:, 1], points[:, 0]] = encoded(points)

	# cuda0 = torch.device('cuda:0')
	# tensor = torch.tensor([0, 0], dtype=torch.short, device=cuda0)

	print(points)
	print(arr)
	# print(tensor)

	# Compute 3D Voronoi diagram
	# Input: a 3D texture. Each pixel is an integer encoding 3 coordinates. 
	# 		For each site at (x, y, z), the pixel at coordinate (x, y, z) should contain 
	# 		the encoded coordinate (x, y, z). Pixels that are not sites should contain 
	# 		the integer MARKER. Use ENCODE (and DECODE) macro to encode (and decode).
	# See our website for the effect of the three parameters: 
	# 		phase1Band, phase2Band, phase3Band
	# Parameters must divide textureSize
	pba3d.voronoi(arr, 1, 1, 2)

	print(arr)

	return arr

def voronoi_to_dist(voronoi):
	""" voronoi is encoded """
	def decoded_nonstacked(p):
		return np.right_shift(p, 20) & 1023, np.right_shift(p, 10) & 1023, p & 1023

	x_i, y_i, z_i = np.indices(voronoi.shape)
	x_v, y_v, z_v = decoded_nonstacked(voronoi)

	return np.sqrt((x_v - x_i) ** 2 + (y_v - y_i) ** 2 + (z_v - z_i) ** 2)


arr = np.load("test.npy")
dist = voronoi_to_dist(arr)

print(dist)
