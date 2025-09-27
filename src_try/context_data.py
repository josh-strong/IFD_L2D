import torch
from attrdict import AttrDict
import random
from collections import defaultdict
import numpy as np

import numpy
import torch
from attrdict import AttrDict
import random
from collections import defaultdict
import numpy as np
import pandas as pd
from train_expert_clf import train_and_compute_prototypicality, TrainableClassifier
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from tqdm import tqdm

class ContextSampler():
    def __init__(self, config, context_images, context_labels, query_images, query_labels, experts_lst, n_cntx_pts, context_expert_preds, query_expert_preds, seed=None, device='cpu', **kwargs):
        self.n_cntx_pts = n_cntx_pts
        self.device = device
        self.num_data = len(context_labels)
        self.seed = seed
        self.config = config
        self.dataset  = config.dataset
        if self.dataset == 'ham10000':
            self.CLASS_NAMES = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']


        self.context_images = context_images
        self.context_labels = context_labels
        self.context_expert_preds = context_expert_preds
    
        self.query_images = query_images
        self.query_labels = query_labels
        self.query_expert_preds = query_expert_preds

        self.experts_lst = experts_lst
        self.n_experts = len(experts_lst)

        # Convert numpy arrays to tensors if needed
        if not torch.is_tensor(self.context_labels):
            self.context_labels = torch.tensor(self.context_labels)
        if not torch.is_tensor(self.query_labels):
            self.query_labels = torch.tensor(self.query_labels)
        if not torch.is_tensor(self.context_images):
            self.context_images = torch.tensor(self.context_images)
        if not torch.is_tensor(self.query_images):
            self.query_images = torch.tensor(self.query_images)
            
        self.cntx = AttrDict()
        self.cntx.xc = self.context_images.unsqueeze(0).repeat(self.n_experts,1,1,1,1)
        self.cntx.yc = self.context_labels.unsqueeze(0).repeat(self.n_experts,1)

        self.cntx.mc = self.context_expert_preds
        if self.n_experts == 1:
            self.cntx.mc = self.cntx.mc.unsqueeze(0)

        self.query = AttrDict()
        self.query.xc = self.query_images.unsqueeze(0).repeat(self.n_experts,1,1,1,1)
        self.query.yc = self.query_labels.unsqueeze(0).repeat(self.n_experts,1)

        self.query.mc = self.query_expert_preds
        if self.n_experts == 1:
            self.query.mc = self.query.mc.unsqueeze(0)
    
    def sample(self):
        random_idxs = random.sample(range(self.num_data), self.n_cntx_pts)
        sampled_cntx = AttrDict()
        sampled_cntx.xc = self.cntx.xc[:,random_idxs,:,:,:]
        sampled_cntx.yc = self.cntx.yc[:,random_idxs]
        sampled_cntx.mc = self.cntx.mc[:,random_idxs]

        return sampled_cntx
    
    def send_context_to_device(self):
        self.cntx.xc = self.cntx.xc.to(self.device)
        self.cntx.yc = self.cntx.yc.to(self.device)
        self.cntx.mc = self.cntx.mc.to(self.device)

    
def get_context_data(config, train_data, val_data, test_data=None):
    # the only random splitting we do for ham is on training data into query + context (dependent on seed of experiment)
    # (not for valid or test)
    samples_per_class = 15
    class_indices = defaultdict(list)
    for idx, target in enumerate(train_data.targets):
        class_indices[target.item()].append(idx)
    generator = torch.Generator().manual_seed(config["seed"])

    sampled_indices = []
    for class_label, indices in class_indices.items():
        shuffled_indices = torch.randperm(len(indices), generator=generator)[:samples_per_class]
        selected_indices = [indices[i] for i in shuffled_indices]
        sampled_indices.extend(selected_indices)

    all_indices = np.arange(len(train_data))
    query_indices = torch.tensor(list(set(all_indices) - set(sampled_indices)))
    cntx_indices = torch.tensor(sampled_indices)
    assert len(cntx_indices) + len(query_indices) == len(all_indices)
    assert set(query_indices.numpy()) | set(cntx_indices.numpy()) == set(all_indices)
    X_c = train_data.data[cntx_indices]
    y_c = train_data.targets[cntx_indices]

    X_tr_q = train_data.data[query_indices]
    y_tr_q = train_data.targets[query_indices]

    X_val_q = val_data.data
    y_val_q = val_data.targets

    if test_data is None:
        return X_c, y_c, X_tr_q, y_tr_q, X_val_q, y_val_q, (cntx_indices, query_indices)
    else:
        X_tst_q = test_data.data
        y_tst_q = test_data.targets
        return X_c, y_c, X_tr_q, y_tr_q, X_tst_q, y_tst_q, (cntx_indices, query_indices)


### NEW STUFF

def get_sigmoid_params(acc_easy, acc_hard):
    """
    Calculates the alpha and beta parameters for the sigmoid function
    based on desired accuracies at the easiest and hardest prototypicality points.
    
    Our prototypicality scores are in range [0, 1] where:
    - proto = 1.0 for most prototypical cases (easy)
    - proto = 0.0 for least prototypical cases (hard)
    
    Args:
        acc_easy: Target accuracy for prototypical cases (proto=1.0, e.g., 0.99)
        acc_hard: Target accuracy for atypical cases (proto=0.0, e.g., 0.90)
    
    Returns:
        alpha, beta: Parameters for sigmoid function σ(α * proto + β)
    """
    # Inverse sigmoid (logit) function: logit(p) = -log(1/p - 1)
    logit = lambda p: -np.log(1/p - 1)
    
    # Ensure probabilities are not exactly 0 or 1 to avoid log errors
    epsilon = 1e-7
    acc_easy = np.clip(acc_easy, epsilon, 1 - epsilon)
    acc_hard = np.clip(acc_hard, epsilon, 1 - epsilon)
    
    # Calculate parameters:
    # At proto=1 (easy): σ(α + β) = acc_easy  => α + β = logit(acc_easy)
    # At proto=0 (hard): σ(β) = acc_hard      => β = logit(acc_hard)
    # Therefore: α = logit(acc_easy) - logit(acc_hard)
    
    beta = logit(acc_hard)
    alpha = logit(acc_easy) - logit(acc_hard)
    
    return alpha, beta


alpha_spec, beta_spec = get_sigmoid_params(acc_easy=0.999, acc_hard=0.70)

# Non-Specialty Class: Moderate instance-dependent curve from 50% to 35%
# This shows instance dependence but much more moderate than specialty
alpha_ns, beta_ns = get_sigmoid_params(acc_easy=0.50, acc_hard=0.35)


# We only need one archetype now
UNIFIED_EXPERT_ARCHETYPE = {
    'name': 'Specialist with Graceful Degradation',
    'params': {
        'alpha_specialty': alpha_spec,
        'beta_specialty': beta_spec,
        'alpha_non_specialty': alpha_ns,
        'beta_non_specialty': beta_ns,
    }
}


def simulate_prediction(config, image, true_label, expert_params, class_centroids, max_distances, trained_classifier):
        """
        Simulate an expert's prediction for a single image.
        
        Args:
            image: Input image tensor
            true_label: Ground truth label
            expert_params: Dictionary containing expert configuration
            class_centroids: Class centroids from simulation setup
            max_distances: Max distances for normalization
            trained_classifier: Trained feature extractor model
        
        Returns:
            Dictionary with prediction results
        """
        trained_classifier.eval()
        with torch.no_grad():
            # Extract features using trained classifier
            if image.dim() == 3:  # Add batch dimension if needed
                image = image.unsqueeze(0)
            
            image = image.to(device)
            features = trained_classifier.extract_features(image)
            
            # Compute prototypicality score
            proto_score = get_prototypicality(features[0], true_label, class_centroids, max_distances)
            
            # Determine if this is the expert's specialty
            is_specialty = true_label in [expert_params['specialty_class']]
            
            # Get appropriate parameters
            if is_specialty:
                alpha = expert_params['alpha_specialty']
                beta = expert_params['beta_specialty']
            else:
                alpha = expert_params['alpha_non_specialty']
                beta = expert_params['beta_non_specialty']
            
            # Compute instance-specific accuracy
            accuracy = get_instance_accuracy(proto_score, alpha, beta)
            
            # Generate stochastic prediction
            is_correct = np.random.random() < accuracy
            
            if is_correct:
                prediction = true_label
            else:
                # Random incorrect prediction
                possible_labels = [i for i in range(config.n_classes) if i != true_label]
                prediction = np.random.choice(possible_labels)
            
            return {
                'true_label': true_label,
                'prediction': prediction,
                'is_correct': is_correct,
                'proto_score': proto_score,
                'accuracy': accuracy,
                'is_specialty': is_specialty
            }

def create_unified_expert_profile(specialty_class_idx):
    """
    Create an expert profile using the unified expert archetype.
    Each expert follows the same behavioral pattern but specializes in different classes.
    """
    return {
        'name': f'Specialist (Class {specialty_class_idx})',
        'specialty_class': specialty_class_idx,
        'alpha_specialty': UNIFIED_EXPERT_ARCHETYPE['params']['alpha_specialty'],
        'beta_specialty': UNIFIED_EXPERT_ARCHETYPE['params']['beta_specialty'],
        'alpha_non_specialty': UNIFIED_EXPERT_ARCHETYPE['params']['alpha_non_specialty'],
        'beta_non_specialty': UNIFIED_EXPERT_ARCHETYPE['params']['beta_non_specialty'],
        'archetype': 'Unified_Specialist'
    }

def generate_expert_predictions_batch(config, expert_profile, evaluation_data, evaluation_labels, evaluation_indices, batch_size=100, class_centroids=None, max_distances=None, best_state=None):
    """Generate predictions for an expert on the full evaluation dataset."""
    
    all_results = []
    num_samples = len(evaluation_data)
    
    # Process in batches for efficiency
    for start_idx in tqdm(range(0, num_samples, batch_size), 
                         desc=f"Predicting class {expert_profile['name']}"):
        
        end_idx = min(start_idx + batch_size, num_samples)
        batch_results = []
        
        for i in range(start_idx, end_idx):
            image = evaluation_data[i]
            true_label = evaluation_labels[i].item()

            exp_clf = TrainableClassifier(config.n_classes, config.dataset)
            exp_clf.to(device)
            exp_clf.load_state_dict(best_state)
            
            # Generate prediction
            result = simulate_prediction(
                config, image, true_label, expert_profile,
                class_centroids, max_distances, exp_clf
            )
            
            # Add sample index
            result['sample_idx'] = i
            batch_results.append(result)
        
        all_results.extend(batch_results)
    
    return all_results

import torch
import numpy as np
from train_expert_clf import TrainableClassifier


def get_prototypicality(features, true_label, class_centroids, max_distances):
    """
    Calculate prototypicality score for an instance.
    
    Args:
        features: Feature vector for the instance
        true_label: Ground truth class label
        class_centroids: Dict mapping class indices to centroid features
        max_distances: Dict mapping class indices to maximum distances
    
    Returns:
        Prototypicality score between 0 and 1
    """
    true_label = true_label.item() if torch.is_tensor(true_label) else true_label
    
    if true_label not in class_centroids:
        return 0.5  # Default for unknown classes
    
    # Calculate distance to class centroid
    centroid = class_centroids[true_label]
    
    # Ensure both tensors are on the same device
    if features.device != centroid.device:
        centroid = centroid.to(features.device)
    
    distance = torch.norm(features - centroid).item()
    
    # Normalize by maximum distance for this class
    max_dist = max_distances[true_label]
    normalized_distance = min(distance / max_dist, 1.0) if max_dist > 0 else 0.0
    
    # Convert distance to prototypicality (closer = more prototypical)
    prototypicality = 1.0 - normalized_distance
    
    return prototypicality

def get_instance_accuracy(proto_score, alpha, beta):
    """
    Compute instance-specific accuracy based on prototypicality score.
    
    Args:
        proto_score: Prototypicality score between 0 and 1
        alpha: Steepness parameter for sigmoid curve
        beta: Shift parameter for sigmoid curve
    
    Returns:
        Accuracy probability between 0 and 1
    """
    # Sigmoid function: σ(α * proto + β)
    z = alpha * proto_score + beta
    accuracy = 1.0 / (1.0 + np.exp(-z))
    return min(1.0, max(0.0, accuracy))

def simulate_prediction(config, image, true_label, expert_params, class_centroids, max_distances, trained_classifier):
        """
        Simulate an expert's prediction for a single image.
        
        Args:
            image: Input image tensor
            true_label: Ground truth label
            expert_params: Dictionary containing expert configuration
            class_centroids: Class centroids from simulation setup
            max_distances: Max distances for normalization
            trained_classifier: Trained feature extractor model
        
        Returns:
            Dictionary with prediction results
        """
        trained_classifier.eval()
        with torch.no_grad():
            # Extract features using trained classifier
            if image.dim() == 3:  # Add batch dimension if needed
                image = image.unsqueeze(0)
            
            image = image.to(device)
            features = trained_classifier.extract_features(image)
            
            # Compute prototypicality score
            proto_score = get_prototypicality(features[0], true_label, class_centroids, max_distances)
            
            # Determine if this is the expert's specialty
            is_specialty = true_label in [expert_params['specialty_class']]
            
            # Get appropriate parameters
            if is_specialty:
                alpha = expert_params['alpha_specialty']
                beta = expert_params['beta_specialty']
            else:
                alpha = expert_params['alpha_non_specialty']
                beta = expert_params['beta_non_specialty']
            
            # Compute instance-specific accuracy
            accuracy = get_instance_accuracy(proto_score, alpha, beta)
            
            # Generate stochastic prediction
            is_correct = np.random.random() < accuracy
            
            if is_correct:
                prediction = true_label
            else:
                # Random incorrect prediction
                possible_labels = [i for i in range(config.n_classes) if i != true_label]
                prediction = np.random.choice(possible_labels)
            
            return {
                'true_label': true_label,
                'prediction': prediction,
                'is_correct': is_correct,
                'proto_score': proto_score,
                'accuracy': accuracy,
                'is_specialty': is_specialty
            }
