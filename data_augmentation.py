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
    # Paramètres de transformation affine ajustés
    angle1 = np.random.uniform(-10, 10)
    angle2 = 0
    translation = np.random.uniform(-0.05, 0.05, size=3)
    scale = np.random.uniform(0.95, 1.05)

    matrix1 = torch.tensor([
        [scale * np.cos(np.radians(angle1)), -np.sin(np.radians(angle1)), 0, translation[0]],
        [np.sin(np.radians(angle1)), scale * np.cos(np.radians(angle1)), 0, translation[1]],
        [0, 0, scale, translation[2]]
    ], dtype=torch.float32)

    matrix2 = torch.tensor([
        [scale * np.cos(np.radians(angle2)), -np.sin(np.radians(angle2)), 0, translation[0]],
        [np.sin(np.radians(angle2)), scale * np.cos(np.radians(angle2)), 0, translation[1]],
        [0, 0, scale, translation[2]]
    ], dtype=torch.float32)

    # Ajoutez une dimension pour N (le batch)
    matrix1 = matrix1.unsqueeze(0)
    matrix2 = matrix2.unsqueeze(0)

    # Appliquer la transformation affine
    affine_grid1 = F.affine_grid(matrix1, image_tensor1.size(), align_corners=False)
    transformed_image1 = F.grid_sample(image_tensor1, affine_grid1, align_corners=False)
    
    affine_grid2 = F.affine_grid(matrix2, image_tensor2.size(), align_corners=False)
    transformed_image2 = F.grid_sample(image_tensor2, affine_grid2, align_corners=False)

    # Vérifiez les valeurs après transformation affine
    print(f"Affine transformed_image1 min: {transformed_image1.min().item()}, max: {transformed_image1.max().item()}")
    print(f"Affine transformed_image2 min: {transformed_image2.min().item()}, max: {transformed_image2.max().item()}")

    # Normaliser les images transformées
    transformed_image1 = torch.clamp(transformed_image1, 0, 1)
    transformed_image2 = torch.clamp(transformed_image2, 0, 1)

    # Déformation élastique
    alpha = np.random.uniform(100, 300)
    sigma = np.random.uniform(30, 80)
    random_state = np.random.RandomState(None)
    shape = transformed_image1.shape[-3:]
    dx = ndi.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = ndi.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = ndi.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    indices = np.array([x + dx, y + dy, z + dz]).reshape(3, *shape)

    distorted_image1 = ndi.map_coordinates(transformed_image1.numpy()[0, 0], indices, order=1, mode='reflect')
    distorted_image2 = ndi.map_coordinates(transformed_image2.numpy()[0, 0], indices, order=1, mode='reflect')

    # Normaliser les images distordues
    distorted_image1 = np.clip(distorted_image1, 0, 1)
    distorted_image2 = np.clip(distorted_image2, 0, 1)

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

            image_tensor1 = torch.tensor(image1, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            image_tensor2 = torch.tensor(image2, dtype=torch.float32).unsqueeze(0).unsqueeze(0)       
            
            # Appliquer les transformations synchronisées
            N = 4
            for i in range(N):
                transformed_image1, transformed_image2 = synchronized_transform_3d(image_tensor1, image_tensor2)
                # Sauvegarder les images transforméestransformed_filename1transformed_filename1
                transformed_filename1 = filename + f"_transformed_{i}.tiff"
                transformed_filename2 = filename + f"_transformed_{i}.tiff"
                tiff.imwrite(os.path.join(dir1, transformed_filename1), transformed_image1.squeeze(0).numpy())
                tiff.imwrite(os.path.join(dir2, transformed_filename2), transformed_image2.squeeze(0).numpy())

def delete_transformed_images(dir_path):
    # Parcourir tous les fichiers dans le répertoire donné
    for filename in os.listdir(dir_path):
        # Vérifier si le nom du fichier contient le motif '_transformed_'
        if '_transformed_' in filename:
            file_path = os.path.join(dir_path, filename)
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Could not delete {file_path}: {e}")


if __name__ == "__main__":
    
    dir1 = './pca_based_dataset/testset/target_test/'
    dir2 = './pca_based_dataset/testset/source_test/'
    dir3 = './pca_based_dataset/trainset/source_train/'
    dir4 = './pca_based_dataset/trainset/target_train/'
    apply_transformations_to_pairs(dir1, dir2)
    apply_transformations_to_pairs(dir3,dir4)
