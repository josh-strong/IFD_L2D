import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from resnet224 import ResNet34
from torchvision.models import efficientnet_b0
from tqdm import tqdm
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from helper_fncs import load_dataset

def get_backbone_for_dataset(dataset_name):
    """Get appropriate backbone architecture for the dataset."""
    if dataset_name in ['ham10000', 'new_ham10000', 'cifar10']:
        backbone = ResNet34()
        backbone.n_features = backbone.n_features
    elif dataset_name == 'imagenet16_grey':
        backbone = ResNet34()
        conv1 = backbone.resnet.conv1
        backbone.resnet.conv1 = torch.nn.Conv2d(1, conv1.out_channels, kernel_size=conv1.kernel_size, stride=conv1.stride, padding=conv1.padding, bias=conv1.bias)
        backbone.n_features = backbone.n_features
    elif dataset_name in ['bus', 'organs_axial']:
        backbone = efficientnet_b0(weights=True)
        # Modify first conv layer to accept 1-channel input for grayscale images
        backbone.features[0][0] = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # Add n_features attribute to match ResNet34 interface
        backbone.n_features = backbone.classifier[1].in_features
    elif dataset_name == 'blood_mnist':
        # BloodMNIST: 3-channel RGB images, use EfficientNet
        backbone = efficientnet_b0(weights=True)
        # Keep 3-channel input for RGB images
        backbone.n_features = backbone.classifier[1].in_features
    elif dataset_name == 'oct':
        # OCTMNIST: 1-channel grayscale images, use EfficientNet
        backbone = efficientnet_b0(weights=True)
        # Modify first conv layer to accept 1-channel input for grayscale images
        backbone.features[0][0] = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        backbone.n_features = backbone.classifier[1].in_features
    else:
        # Default to ResNet34 for other datasets
        backbone = ResNet34()
    return backbone

def train_and_compute_prototypicality(config):
    model_path = f"best_expert_clf_{config.dataset}.pth"
    if os.path.exists(model_path):
        print(f"Classifier already trained for dataset '{config.dataset}'. File '{model_path}' exists. Skipping training.")

        # load the best state and the class centroids and max distances
        best_state = torch.load(f"best_expert_clf_{config.dataset}.pth", weights_only=False)['best_state']
        class_centroids = torch.load(f"best_expert_clf_{config.dataset}.pth", weights_only=False)['class_centroids']
        max_distances = torch.load(f"best_expert_clf_{config.dataset}.pth", weights_only=False)['max_distances']
        return best_state, class_centroids, max_distances

    _, _, _, expert_data = load_dataset(config)

    best_state = train_expert_clf(config, expert_data)
    class_centroids, max_distances = compute_prototypicality(config, expert_data, best_state)

    # Save the best state and the class centroids and max distances in a single object
    torch.save({
        'best_state': best_state,
        'class_centroids': class_centroids,
        'max_distances': max_distances
    }, f"best_expert_clf_{config.dataset}.pth")
    print(f"Saved best expert classifier and prototypicality scores to {f'best_expert_clf_{config.dataset}.pth'}")

    # load the best state and the class centroids and max distances
    best_state = torch.load(f"best_expert_clf_{config.dataset}.pth", weights_only=False)['best_state']
    class_centroids = torch.load(f"best_expert_clf_{config.dataset}.pth", weights_only=False)['class_centroids']
    max_distances = torch.load(f"best_expert_clf_{config.dataset}.pth", weights_only=False)['max_distances']

    return best_state, class_centroids, max_distances


class TrainableClassifier(nn.Module):
    """Trainable classifier wrapper for dataset-specific feature extractor."""

    def __init__(self, num_classes, dataset_name):
        super().__init__()
        self.backbone = get_backbone_for_dataset(dataset_name)
        self.dataset_name = dataset_name
        
        # Add dropout for small/medium datasets to prevent overfitting
        if dataset_name == 'organs_axial':
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(self.backbone.n_features, num_classes)
            )
        elif dataset_name == 'blood_mnist':
            self.classifier = nn.Sequential(
                nn.Dropout(0.3),  # Light dropout for medium-sized dataset
                nn.Linear(self.backbone.n_features, num_classes)
            )
        elif dataset_name == 'oct':
            self.classifier = nn.Sequential(
                nn.Dropout(0.4),  # Moderate dropout for small subsampled dataset
                nn.Linear(self.backbone.n_features, num_classes)
            )
        else:
            self.classifier = nn.Linear(self.backbone.n_features, num_classes)
        
        self.n_features = self.backbone.n_features

    def forward(self, x):
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits

    def extract_features(self, x):
        """Extract features without classification head."""
        if hasattr(self.backbone, 'features'):
            # For EfficientNet-based backbones
            x = self.backbone.features(x)
            x = self.backbone.avgpool(x)
            features = torch.flatten(x, 1)
            return features
        else:
            # For ResNet-based backbones
            return self.backbone(x)

def train_expert_clf(config, expert_data):

    expert_indices = list(range(len(expert_data)))
    train_idx, val_idx = train_test_split(
        expert_indices,
        test_size=0.2,
        stratify=[expert_data[i][1] for i in expert_indices],  # assuming __getitem__ returns (x, y)
        random_state=42
    )

    train_data = Subset(expert_data, train_idx)
    val_data = Subset(expert_data, val_idx)


    # === CLASSIFIER TRAINING ===

    print("Training classifier on simulation setup data...")

    print(f"Training set: {len(train_data)} samples")
    print(f"Validation set: {len(val_data)} samples")

    # Dataset-specific hyperparameters
    if config.dataset == 'organs_axial':
        # Organs axial: small dataset (~100 samples), needs aggressive regularization
        batch_size = 16  # Small batch size for small dataset (was 128, causing only 1 batch/epoch)
        learning_rate = 0.0001  # Lower learning rate (was 0.001, too aggressive)
        weight_decay = 1e-4  # L2 regularization (was 0)
        num_epochs = 100  # More epochs but with early stopping
        early_stopping_patience = 15  # Shorter patience for small dataset
        # Also uses: dropout (0.5), learning rate scheduler, modified backbone (EfficientNet)
    elif config.dataset == 'blood_mnist':
        # BloodMNIST: medium-sized dataset, balanced approach
        batch_size = 64  # Medium batch size
        learning_rate = 0.0005  # Moderate learning rate
        weight_decay = 1e-4  # Light regularization
        num_epochs = 75  # More epochs than HAM10000, less than organs_axial
        early_stopping_patience = 20  # Moderate patience
    elif config.dataset == 'oct':
        # OCTMNIST: small subsampled dataset, needs careful tuning
        batch_size = 32  # Small batch size for small dataset
        learning_rate = 0.0003  # Conservative learning rate
        weight_decay = 1e-4  # L2 regularization
        num_epochs = 100  # More epochs for convergence
        early_stopping_patience = 20  # Moderate patience
    elif config.dataset in ['ham10000', 'new_ham10000']:
        # HAM10000/New HAM10000: RGB datasets, original hyperparameters
        batch_size = 128
        learning_rate = 0.001
        weight_decay = 0
        num_epochs = 50
        early_stopping_patience = 25
    elif config.dataset == 'cifar10':
        # CIFAR-10: RGB dataset, memory-efficient hyperparameters
        batch_size = 32  # Further reduced for memory efficiency
        learning_rate = 0.001
        weight_decay = 0
        num_epochs = 50
        early_stopping_patience = 25
    else:
        # Default hyperparameters
        batch_size = 64
        learning_rate = 0.0005
        weight_decay = 1e-5
        num_epochs = 50
        early_stopping_patience = 20

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize model
    trainable_classifier = TrainableClassifier(config.n_classes, config.dataset)
    trainable_classifier.to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(trainable_classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Add learning rate scheduler for smaller datasets
    if config.dataset == 'organs_axial':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    elif config.dataset == 'blood_mnist':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=8, verbose=True)
    elif config.dataset == 'oct':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.6, patience=6, verbose=True)
    else:
        scheduler = None
    
    best_val_acc = 0.0
    patience_counter = 0

    print(f"Starting training with hyperparameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Max epochs: {num_epochs}")
    print(f"  Early stopping patience: {early_stopping_patience}")

    for epoch in range(num_epochs):
        # Training phase
        trainable_classifier.train()
        train_loss = 0.0
        train_correct = 0

        for batch_data, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            outputs = trainable_classifier(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == batch_labels).sum().item()

        train_acc = train_correct / len(train_data)

        # Validation phase
        trainable_classifier.eval()
        val_correct = 0
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                outputs = trainable_classifier(batch_data)
                val_correct += (outputs.argmax(1) == batch_labels).sum().item()

        val_acc = val_correct / len(val_data)

        print(f"Epoch {epoch+1}: Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}")

        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(val_acc)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_state = trainable_classifier.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Training complete! Best validation accuracy: {best_val_acc:.3f}")
    print("✓ Classifier ready for feature extraction!")

    return best_state 


def compute_prototypicality(config, expert_data, best_state):
    # === PROTOTYPICALITY COMPUTATION ===

    print("Computing class centroids and prototypicality scores...")

    # Compute class centroids using simulation setup data
    class_centroids = {}
    max_distances = {}

    setup_loader = DataLoader(expert_data, batch_size=128, shuffle=False, num_workers=2)

    # Collect features for each class
    from collections import defaultdict
    class_features = defaultdict(list)

    trainable_classifier = TrainableClassifier(config.n_classes, config.dataset)
    trainable_classifier.to(device)
    trainable_classifier.load_state_dict(best_state)
    trainable_classifier.eval()

    print("Extracting features for centroid computation...")
    with torch.no_grad():
        for batch_data, batch_labels in tqdm(setup_loader, desc="Extracting features"):
            batch_data = batch_data.to(device)
            features = trainable_classifier.extract_features(batch_data)
            
            for feature, label in zip(features.cpu(), batch_labels):
                class_features[label.item()].append(feature)

    # Compute centroids and max distances
    for class_idx in range(config.n_classes):
        if class_idx in class_features:
            features = torch.stack(class_features[class_idx])
            centroid = features.mean(dim=0)
            
            # Compute max distance for normalization
            distances = torch.norm(features - centroid.unsqueeze(0), dim=1)
            max_dist = distances.max().item()
            
            class_centroids[class_idx] = centroid
            max_distances[class_idx] = max_dist
    
    print("✓ Class centroids computed!")
    return class_centroids, max_distances