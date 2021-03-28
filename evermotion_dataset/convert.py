
from glob import glob
import h5py
import numpy as np
from PIL import Image
from shutil import copyfile
from copy import copy

semantic_paths = glob("./**/frame*.semantic.hdf5", recursive=True)
semantic_paths.sort()

i = 0
for semantic_path in semantic_paths:
    try:
        with h5py.File(semantic_path, "r") as f:
            semantic_id = f["dataset"][:].astype(np.uint8)
    except:
        print(f'failed semantic open: {semantic_path}')
        continue

    try:
        image_path = copy(semantic_path).replace('.semantic.hdf5', '.tonemap.jpg').replace('_geometry_hdf5', '_final_hdf5')
        image = Image.open(image_path)
    except:
        print(f'failed image open: {image_path}')
        continue

    try:
        normals_path = copy(semantic_path).replace('.semantic.hdf5', '.normal_cam.hdf5')
        with h5py.File(normals_path, "r") as f:
            normals = f["dataset"][:].astype(np.half)
    except:
        print(f'failed normals open: {normals_path}')
        continue

    semantic_id[semantic_id == 255] = 40
    semantic_id_image = Image.fromarray(semantic_id)
    semantic_id_image = semantic_id_image.resize((semantic_id_image.size[0] // 2, semantic_id_image.size[1] // 2), resample=Image.NEAREST)
    semantic_id_image.save(f'./train/label/{i:06d}.png')

    normals = (normals * 0.5 + 0.5) * 255
    normals = normals.astype(np.uint8)
    normals_image = Image.fromarray(normals, 'RGB')
    normals_image = normals_image.resize((normals_image.size[0] // 2, normals_image.size[1] // 2), resample=Image.NEAREST)
    normals_image.save(f'./train/normals/{i:06d}.png')

    image = image.resize((image.size[0] // 2, image.size[1] // 2))
    image.save(f'./train/img/{i:06d}.jpg')
    # copyfile(image_path, f'./train/img/{i:06d}.jpg')
    i += 1
    if i % 10000 == 0:
        print("another 10k passed")