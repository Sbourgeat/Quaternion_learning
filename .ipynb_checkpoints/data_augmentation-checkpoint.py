import os
import numpy as np
import tifffile as tiff
from skimage.transform import resize
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scipy.ndimage as ndi

def synchronized_transform_3d(image_tensor1, image_tensor2):
    
    # Paramètres aléatoires pour la transformation affine
    angle1 = np.random.uniform(-25, 25)  # Rotation aléatoire en degrés
    angle2 = 0
    translation = np.random.uniform(-0.1, 0.1, size=3)  # Translation aléatoire
    scale = np.random.uniform(0.9, 1.1)  # Échelle aléatoire
    shear = np.random.uniform(-10, 10)  # Cisaillement aléatoire

    # Matrice de transformation affine 3D aléatoire avec angle
    matrix1 = torch.tensor([[
        [scale * np.cos(np.radians(angle1)), -np.sin(np.radians(angle1)), shear, translation[0]],
        [np.sin(np.radians(angle1)), scale * np.cos(np.radians(angle1)), shear, translation[1]],
        [shear, shear, scale, translation[2]]
    ]], dtype=torch.float32)

    # Matrice de transformation affine 3D aléatoire sans angle
    matrix2 = torch.tensor([[
        [scale * np.cos(np.radians(angle2)), -np.sin(np.radians(angle2)), shear, translation[0]],
        [np.sin(np.radians(angle2)), scale * np.cos(np.radians(angle2)), shear, translation[1]],
        [shear, shear, scale, translation[2]]
    ]], dtype=torch.float32)

    # Appliquer la transformation affine aux deux images
    affine_grid1 = F.affine_grid(matrix1, image_tensor1.size(), align_corners=False)
    transformed_image1 = F.grid_sample(image_tensor1, affine_grid1, align_corners=False)
    
    affine_grid2 = F.affine_grid(matrix2, image_tensor2.size(), align_corners=False)
    transformed_image2 = F.grid_sample(image_tensor2, affine_grid2, align_corners=False)

    # Paramètres aléatoires pour la déformation élastique
    alpha = np.random.uniform(100, 300)  # Déformation moyenne
    sigma = np.random.uniform(30, 80)  # Douceur de la déformation

    # Créer des déplacements aléatoires
    random_state = np.random.RandomState(None)
    shape = transformed_image1.shape[-3:]
    dx = ndi.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval = 0) * alpha
    dy = ndi.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval = 0) * alpha
    dz = ndi.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval = 0) * alpha

    # Appliquer les déplacements aux deux images
    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing = 'ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(z + dz, (-1, 1))
    
    distorted_image1 = ndi.map_coordinates(transformed_image1.numpy()[0], indices, order=1, mode='reflect').reshape(shape)
    distorted_image2 = ndi.map_coordinates(transformed_image2.numpy()[0], indices, order=1, mode='reflect').reshape(shape)

    # Retourner les images transformées sous forme de tenseurs PyTorch
    return torch.tensor(distorted_image1, dtype=torch.float32).unsqueeze(0), torch.tensor(distorted_image2, dtype=torch.float32).unsqueeze(0)

def apply_transformations_to_pairs(dir1, dir2):
    # Parcourir tous les fichiers dans le premier répertoire

    for filename in os.listdir(dir1):
        if filename in os.listdir(dir2): 
            # Charger les deux images
            image_path1 = os.path.join(dir1, filename)
            image_path2 = os.path.join(dir2, filename)
            
            image1 = tiff.imread(image_path1)
            image2 = tiff.imread(image_path2)
            
            image_tensor1 = torch.tensor(image1, dtype=torch.float32).unsqueeze(0)
            image_tensor2 = torch.tensor(image2, dtype=torch.float32).unsqueeze(0)
            
            # Appliquer les transformations synchronisées
            N = 4
            for i in range(N):
                transformed_image1, transformed_image2 = synchronized_transform_3d(image_tensor1, image_tensor2)
                # Sauvegarder les images transformées
                filename  = filename + f"_transformed_{i}" 
                tiff.imwrite(os.path.join(dir1, filename), transformed_image1.squeeze(0).numpy())
                tiff.imwrite(os.path.join(dir2, filename), transformed_image2.squeeze(0).numpy())



if __name__ == "__main__":
    
    dir1 = 'path_to_directory1'
    dir2 = 'path_to_directory2'
    apply_transformations_to_pairs(dir1, dir2)