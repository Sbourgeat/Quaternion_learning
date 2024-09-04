#!/usr/bin/env python
# coding: utf-8

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
import scipy.ndimage as nd
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize
from sklearn.decomposition import PCA

def load_images(directory_path, target_shape):
    images = []
    target_shape = (target_shape, target_shape, target_shape)

    for filename in sorted(os.listdir(directory_path)):
        if filename.endswith(".tif"):
            image_path = os.path.join(directory_path, filename)

            try:
                # Chargement de l'image
                image = tiff.imread(image_path)
                print(f"Loaded {filename} with shape {image.shape} and dtype {image.dtype}")

                # Vérifie si l'image est vide ou corrompue
                if image is None or image.size == 0:
                    print(f"Warning: {filename} is empty or corrupted.")
                    continue

                # Vérifie que l'image a bien été chargée en 3D
                if len(image.shape) != 3:
                    print(f"Warning: {filename} is not a 3D image. Skipping.")
                    continue

                # Redimensionnement si nécessaire
                if image.shape != target_shape:
                    print(f"Resizing {filename} from shape {image.shape} to target shape {target_shape}.")
                    image = resize(image, target_shape, preserve_range=True, anti_aliasing=True)
                    image = image.astype(np.float32)

                # Vérifie si l'image a été redimensionnée correctement
                if image.shape != target_shape:
                    print(f"Warning: {filename} could not be resized properly to {target_shape}. Skipping.")
                    continue

                # Ajout de l'image à la liste
                images.append(image)

            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue

    # Conversion en tableau NumPy
    images = np.array(images)

    # Vérifie si des images ont été chargées
    if len(images) == 0:
        print("No images were loaded successfully. Check the input directory or file formats.")
        return images

    # Ajouter la dimension de canal
    images = np.expand_dims(images, axis=-1)

    print(f"Total images loaded: {len(images)}")
    return images


def normalize(source_images):
    
    normalized_images = []

    for img in source_images:
        min_val = np.min(img)
        max_val = np.max(img)
        normalized_img = (img - min_val) / (max_val - min_val + np.finfo(float).eps)
        normalized_images.append(normalized_img)
    normalized_images = np.array(normalized_images)
    normalized_images = np.clip(normalized_images, 0, 1)
    normalized_images = normalized_images.astype(np.float32)

    return normalized_images


def binarize_targets(target_path, threshold = 0.005):
    
    binarized_image = []

    for targets in target_path:
        targets[targets >= threshold] = 1
        targets[targets < threshold] = 0
        binarized_image.append(targets)
    targets = np.array(binarized_image)
    targets = np.clip(targets, 0, 1)
    targets = targets.astype(np.float32)

    return targets

def quat_finder(images):
    quaternions = []

    for image in images:
        # Convertir le tenseur sur le CPU
        image = image.cpu().detach()  # Assurez-vous qu'il n'y a pas de gradient attaché
        activated_coords = np.array(np.where(image > 0)).T[:, :3]  # Ne garder que les 3 premières dimensions
        num_activated_points = activated_coords.shape[0]
        print(f"Number of activated points: {num_activated_points}")

        # Vérifier s'il y a suffisamment de points pour la PCA
        if num_activated_points < 3:
            print("Not enough points for PCA. Skipping this image.")
            # Ajouter un quaternion par défaut ou une autre méthode de traitement
            quaternions.append(np.array([1, 0, 0, 0]))  # Quaternion neutre par exemple
            continue

        # Définir n_components de manière dynamique
        n_components = min(3, num_activated_points, activated_coords.shape[1])
        
        # PCA avec le nombre de composantes dynamiquement ajusté
        pca = PCA(n_components=n_components)
        pca.fit(activated_coords)

        # Obtenir les composantes principales
        principal_components = pca.components_

        # Vérification de la forme des composantes principales
        if principal_components.shape != (3, 3):
            print(f"Warning: PCA did not return a (3, 3) matrix. Received shape: {principal_components.shape}")
            # Ajouter un quaternion par défaut ou une autre méthode de traitement
            quaternions.append(np.array([1, 0, 0, 0]))  # Quaternion neutre par exemple
            continue

        # Utiliser les composantes principales pour définir une matrice de rotation
        rotation_matrix = principal_components.T
        rotation_quaternion = R.from_matrix(rotation_matrix).as_quat()
        quaternions.append(rotation_quaternion)

    # Convertir les quaternions en tenseurs PyTorch
    quaternions = torch.from_numpy(np.array(quaternions)).float()
    quaternions.requires_grad = True  # S'assurer que requires_grad est True
    
    return quaternions


class CustomDataset(Dataset):
    def __init__(
        self, source_images, target_images, downsampling_factor = 1, augment=False
    ):
        self.source_images = torch.tensor(source_images, dtype=torch.float32)
        self.source_images = self.source_images.permute(0, 4, 1, 2, 3)
        self.source_images = torch.nn.functional.interpolate(
            self.source_images, scale_factor=1 / downsampling_factor
        )
        self.source_images = self.source_images.permute(0, 2, 3, 4, 1)
        self.target_images = torch.tensor(target_images, dtype=torch.float32)
        self.target_images = self.target_images.permute(0, 4, 1, 2, 3)
        self.target_images = torch.nn.functional.interpolate(
            self.target_images, scale_factor=1 / downsampling_factor
        )
        self.target_images = self.target_images.permute(0, 2, 3, 4, 1)
        self.augment = augment

    def __len__(self):
        return len(self.source_images)

    def shape(self):
        return self.source_images.shape

    def __getitem__(self, idx):
        source, target = self.source_images[idx], self.target_images[idx]
        if self.augment:
            sample = tio.Subject(
                source=tio.ScalarImage(tensor=source),
                target=tio.LabelMap(tensor=target),
            )
            sample = self.transforms(sample)
            source, target = sample.source.tensor, sample.target.tensor
        return source, target



# Définition du modèle

class OrientationNetQuaternion(nn.Module):
    
    def __init__(self):
        super(OrientationNetQuaternion, self).__init__()
        # 3D convolutional layers
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 33 * 33 * 33, 512)  # Remplacez cette taille par la taille correcte calculée dynamiquement
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 4)  # Output 4 quaternions

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)
        x = F.relu(self.conv1(x))
        x = F.max_pool3d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool3d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool3d(x, 2)
        x = x.reshape(x.size(0), -1)  # Aplatir le tenseur de manière dynamique
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Output the quaternion
        return x


# Fonction d'entraînement

def train_model(model, criterion, optimizer, train_loader, num_epochs=10):
    
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0

        for volumes, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(volumes)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Fonction de test

def test_model(model, test_loader):
    
    model.eval()
    total_loss = 0.0
    with torch.no_grad():

        for volumes, labels in test_loader:
            outputs = model(volumes)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    print(f'Average test loss: {total_loss/len(test_loader):.4f}')


class Config:
    pass

if __name__ == "__main__":

    
    # Hard-coded configuration
    config = Config()
    config.model_path = "./pca_based_dataset/models/"  # Change to your model path
    config.input_dir = "./pca_based_dataset/input/"  # Change to your input images directory
    config.output_dir = "./pca_based_dataset/ouput/"  # Change to your output directory
    config.batch_size = 3
    config.target_shape = 264
    config.num_epochs = 500
    config.learning_rate = 1e-3




    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_source_dir = "./pca_based_dataset/trainset/source_train/"  # "/home/sbourgeat/Project/MachineLearning/UNET/cropped_training/cropped_training/source"
    train_target_dir = "./pca_based_dataset/trainset/target_train/"  # "/home/sbourgeat/Project/MachineLearning/UNET/cropped_training/cropped_training/target_binarized"
    test_source_dir = "./pca_based_dataset/testset/source_test/" #  "/home/sbourgeat/Project/MachineLearning/UNET/cropped_test/cropped_test/source"
    test_target_dir = "./pca_based_dataset/testset/target_test/" #"/home/sbourgeat/Project/MachineLearning/UNET/cropped_test/cropped_test/target_binarized"


    # Load datasets
    
    train_source = load_images(train_source_dir, config.target_shape)  
    train_target = load_images(train_target_dir, config.target_shape)
    test_source = load_images(test_source_dir, config.target_shape)
    test_target = load_images(test_target_dir, config.target_shape)
    train_source = normalize(train_source)
    test_source = normalize(test_source)
    train_dataset = CustomDataset(
        train_source, train_target)
    test_dataset = CustomDataset(
        test_source, test_target)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    print("length of training data: ", len(train_source))
    print("length of test data: ", len(test_source))
    print("shape of training data: ", train_source.shape, train_target.shape)
    print("shape of test data: ", test_source.shape, test_target.shape)

    ### MODEL CREATION #################################################################################
    
    # MODEL CREATION
    model = OrientationNetQuaternion()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Définir la fonction de perte
    criterion = nn.MSELoss()  # Par exemple, pour une régression de quaternions

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Early stopping parameters
    patience = 20
    best_val_loss = float("inf")
    early_stop_counter = 0
    
    # Training and evaluation loop

    def evaluate(model, dataloader, criterion, device, epoch):
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():

            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                q_target = quat_finder(targets)
                q_output = quat_finder(outputs)
                loss = criterion(q_output, q_target)
                val_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0.5).float()
                correct += (predicted == targets).sum().item()
                total += targets.numel()
                dice_score += dice_coefficient(predicted, targets) * inputs.size(0)
        val_loss /= len(dataloader.dataset)
        accuracy = correct / total
        dice_score /= len(dataloader.dataset)
        
        return val_loss, accuracy, dice_score


    for epoch in range(config.num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        with tqdm(
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{config.num_epochs}",
            unit="batch",
        ) as pbar:
            for inputs, targets in train_loader:
                # Transfert vers l'appareil GPU/CPU
                inputs, targets = inputs.to(device), targets.to(device)

                # Prédiction du modèle
                outputs = model(inputs)

                # Affichez la forme de `outputs` après la prédiction
                print(f"Shape of outputs after model prediction: {outputs.shape}")
                
                # Ajuster le redimensionnement de 'outputs' en fonction de sa forme réelle

                if outputs.dim() == 2:
                    # Cas où 'outputs' est aplati avec deux dimensions (batch_size, num_features)
                    print(f"Flattened shape of outputs: {outputs.shape}")

                    # Calculez le nombre total d'éléments
                    total_elements = outputs.size(0) * outputs.size(1)
                    expected_elements = config.target_shape * config.target_shape * config.target_shape
    
                    # Vérifiez que le nombre total d'éléments correspond
                    if total_elements == expected_elements:
                        # Effectuez le redimensionnement si le nombre d'éléments est correct
                        outputs = outputs.view(-1, 1, config.target_shape, config.target_shape, config.target_shape)
                    else:
                        print(f"Cannot reshape outputs from size {outputs.size()} to target shape {config.target_shape}.")
                        continue

                elif outputs.dim() == 4:
                    # Si 'outputs' a 4 dimensions (N, D, H, W), ajoutez une dimension de canal
                    outputs = outputs.unsqueeze(1)  # Ajoute une dimension de canal pour obtenir (N, C, D, H, W)

                else:
                    print("Output tensor has unexpected number of dimensions. Check the model's output.")
                    continue


                print(f"Shape of outputs after adjustment: {outputs.shape}")

                # Assurez-vous que 'targets' a également le bon format
                if targets.dim() == 4:
                    targets = targets.unsqueeze(1)  # Ajouter une dimension de canal pour correspondre au format (N, C, D, H, W)

                # Redimensionner les prédictions pour qu'elles correspondent aux cibles
                outputs_resized = F.interpolate(
                    outputs, 
                    size=(targets.shape[2], targets.shape[3], targets.shape[4]), 
                    mode='trilinear', 
                    align_corners=False
                )

                print(f"Shape of resized outputs: {outputs_resized.shape}")

                # Calcul de la perte entre les prédictions redimensionnées et les cibles
                loss = criterion(outputs_resized, targets)

                # Rétropropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Calcul de la précision
                correct += (outputs_resized == targets).sum().item()
                total += targets.numel()

                print(f"Current Accuracy: {correct / total:.4f}")



        # Calcul de la perte d'époque
        epoch_loss = running_loss / len(train_loader.dataset)

        if total > 0:
            epoch_accuracy = correct / total
        else:
            epoch_accuracy = 0
            print("Warning: Total number of targets is zero, accuracy set to 0.")

        val_loss, val_accuracy, val_dice = evaluate(
            model, test_loader, criterion, device, epoch
        )

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), "best_model_3d_aug.pth")

        else:
            early_stop_counter += 1


            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

# Sauvegarder le modèle final après l'entraînement
torch.save(model.state_dict(), "model_3d_aug.pth")
wandb.finish()
