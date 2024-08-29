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
import scipy.ndimage as ndi


def load_images(directory_path, target_shape):
    
    images = []
    target_shape = (target_shape, target_shape, target_shape)

    for filename in sorted(os.listdir(directory_path)):

        if filename.endswith(".tif"):
            image = tiff.imread(os.path.join(directory_path, filename))

            if image.shape != target_shape:
                image = resize(
                    image, target_shape, preserve_range=True, anti_aliasing=True
                )
                image = image.astype(np.float32)
            images.append(image)
    images = np.array(images)
    images = np.expand_dims(images, axis=-1)  # Add channel dimension

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


def binarize_targets(target_path, threshold = 0.1):
    
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
        
        # Step 1: Extract coordinates of non-zero voxels
        activated_coords = np.array(np.where(image > 0)).T
        
        # Step 2: Perform PCA
        pca = PCA(n_components=3)
        pca.fit(activated_coords)
        
        # Get the principal components
        principal_components = pca.components_
        
        # Step 3: Use the principal components to define a rotation quaternion
        rotation_matrix = principal_components.T
        rotation_quaternion = R.from_matrix(rotation_matrix).as_quat()
        quaternions.append(rotation_quaternion)

    quaternions = torch.from_numpy(np.array(quaternions))

    return quaternions

class CustomDataset(Dataset):
    def __init__(self, source_images, downsampling_factor):
        self.source_images = torch.tensor(source_images, dtype=torch.float32)
        
        # Assurez-vous que la forme est correcte avant d'interpoler
        print("Initial shape:", self.source_images.shape)
        
        # Interpolation attend (batch_size, channels, depth, height, width)
        self.source_images = self.source_images.permute(0, 4, 2, 3, 1)  # Reorder to match (N, C, D, H, W)
        
        print("Shape before interpolation:", self.source_images.shape)
        
        self.source_images = torch.nn.functional.interpolate(
            self.source_images, scale_factor=1 / downsampling_factor, mode='trilinear', align_corners=False
        )
        
        print("Shape after interpolation:", self.source_images.shape)
        
        self.source_images = self.source_images.permute(0, 2, 3, 4, 1)  # Revert to original order (N, D, H, W, C)

        def __len__(self):

            return len(self.source_images)

        def __getitem__(self, idx):
            # Retourner l'image à l'index donné
            return self.source_images[idx]


# Définition du modèle

class OrientationNetQuaternion(nn.Module):
    
    def __init__(self):

        super(OrientationNetQuaternion, self).__init__()
        
        # 3D convolutional layers
        
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Fully connected layers
        
        self.fc1 = nn.Linear(64 * 8 * 8 * 8, 512)  # Adjust according to input size
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 4)  # Output 4 quaternions
    
    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.max_pool3d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool3d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool3d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten the tensor
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
    config.batch_size = 5
    config.target_shape = 264
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
    
    model = OrientationNetQuaternion(dropout=config.dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
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
                loss = criterion(q_outputs, q_targets)
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
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                q_target = quat_finder(targets)
                q_output = quat_finder(outputs)
                loss = criterion(q_outputs, q_targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                predicted = outputs
                correct += (predicted == targets).sum().item()
                total += targets.numel()
                pbar.update(1)
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct / total
        val_loss, val_accuracy, val_dice = evaluate(
            model, test_loader, criterion, device, epoch
        )

        # Early stopping

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), "best_model_3d_aug.pth")

        else:
            early_stop_counter += 1

            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    torch.save(model.state_dict(), "model_3d_aug.pth")
    wandb.finish()
    
    # Perform predictions
    
    predictions = []
    with torch.no_grad():

        for inputs in tqdm(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
    
    # Save the predictions
    
    save_predictions(predictions, config.output_dir, original_shape)
    print(f"Predictions saved to {config.output_dir}.")
