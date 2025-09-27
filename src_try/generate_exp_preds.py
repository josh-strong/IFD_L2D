import torch
import torch.nn.functional as F
import numpy as np
import hashlib
import pandas as pd
import json
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from train_expert_clf import TrainableClassifier
from helper_fncs import load_dataset
from utils import set_seed
from collections import defaultdict

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

# Default parameter sets for different expert archetypes
EXPERT_ARCHETYPES = {
    # 'high_specialist': {
    #     'name': 'High Specialist',
    #     'description': 'High accuracy on specialty, moderate on others',
    #     'specialty_params': get_sigmoid_params(acc_easy=0.999, acc_hard=0.70),
    #     'non_specialty_params': get_sigmoid_params(acc_easy=0.50, acc_hard=0.35)
    # },

    # 'new_specialist': {
    #     'name': 'High Specialist',
    #     'description': 'High accuracy on specialty, moderate on others',
    #     'specialty_params': get_sigmoid_params(acc_easy=0.999, acc_hard=0.70),
    #     'non_specialty_params': get_sigmoid_params(acc_easy=0.50, acc_hard=0.35)
    # },
    
    # 'moderate_specialist': {
    #     'name': 'Moderate Specialist', 
    #     'description': 'Good accuracy on specialty, fair on others',
    #     'specialty_params': get_sigmoid_params(acc_easy=0.90, acc_hard=0.60),
    #     'non_specialty_params': get_sigmoid_params(acc_easy=0.45, acc_hard=0.30)
    # },
    # 'generalist': {
    #     'name': 'Generalist',
    #     'description': 'Consistent moderate accuracy across all classes',
    #     'specialty_params': get_sigmoid_params(acc_easy=0.85, acc_hard=0.65),
    #     'non_specialty_params': get_sigmoid_params(acc_easy=0.70, acc_hard=0.50)
    # },
    'variable_specialist': {
        'name': 'Variable Specialist',
        'description': 'High variability based on instance difficulty',
        'specialty_params': get_sigmoid_params(acc_easy=0.99, acc_hard=0.50),
        'non_specialty_params': get_sigmoid_params(acc_easy=0.80, acc_hard=0.20)
    },
    'realistic_specialist': {
        'name': 'Stable Specialist',
        'description': 'Low variability',
        'specialty_params': get_sigmoid_params(acc_easy=0.99, acc_hard=0.85),
        'non_specialty_params': get_sigmoid_params(acc_easy=0.85, acc_hard=0.7)
    },
}

class ExpertRegistry:
    """Registry to manage and identify experts by their parameters and characteristics."""
    
    def __init__(self):
        self.experts = {}  # expert_id -> expert_profile
        self.parameter_index = defaultdict(list)  # (alpha, beta) -> [expert_ids]
        self.specialty_index = defaultdict(list)  # specialty_class -> [expert_ids]
        self.archetype_index = defaultdict(list)  # archetype -> [expert_ids]
        
    def register_expert(self, expert_profile):
        """Register an expert in the registry."""
        expert_id = expert_profile['expert_id']
        self.experts[expert_id] = expert_profile
        
        # Index by parameters
        spec_params = (round(expert_profile['alpha_specialty'], 6), round(expert_profile['beta_specialty'], 6))
        non_spec_params = (round(expert_profile['alpha_non_specialty'], 6), round(expert_profile['beta_non_specialty'], 6))
        
        self.parameter_index[spec_params].append(expert_id)
        self.parameter_index[non_spec_params].append(expert_id)
        
        # Index by specialty
        self.specialty_index[expert_profile['specialty_class']].append(expert_id)
        
        # Index by archetype
        self.archetype_index[expert_profile['archetype']].append(expert_id)
        
    def find_experts_by_parameters(self, alpha, beta, tolerance=1e-6, parameter_type='specialty'):
        """Find experts with specific alpha/beta parameters."""
        matching_experts = []
        for expert_id, expert in self.experts.items():
            if parameter_type == 'specialty':
                expert_alpha = expert['alpha_specialty']
                expert_beta = expert['beta_specialty']
            else:
                expert_alpha = expert['alpha_non_specialty']
                expert_beta = expert['beta_non_specialty']
                
            if (abs(expert_alpha - alpha) < tolerance and 
                abs(expert_beta - beta) < tolerance):
                matching_experts.append((expert_id, parameter_type))
        return matching_experts
    
    def find_experts_by_specialty(self, specialty_class):
        """Find experts specialising in a specific class."""
        return [self.experts[eid] for eid in self.specialty_index[specialty_class]]
    
    def find_experts_by_archetype(self, archetype):
        """Find experts of a specific archetype."""
        return [self.experts[eid] for eid in self.archetype_index[archetype]]
    
    def get_expert_signature(self, expert_id):
        """Get a unique signature for an expert based on their parameters."""
        expert = self.experts[expert_id]
        return {
            'expert_id': expert_id,
            'specialty_class': expert['specialty_class'],
            'archetype': expert['archetype'],
            'specialty_alpha': round(expert['alpha_specialty'], 6),
            'specialty_beta': round(expert['beta_specialty'], 6),
            'non_specialty_alpha': round(expert['alpha_non_specialty'], 6),
            'non_specialty_beta': round(expert['beta_non_specialty'], 6)
        }
    
    def list_all_experts(self):
        """Get summary of all registered experts."""
        return [self.get_expert_signature(eid) for eid in self.experts.keys()]
    
def centroids_to_tensor(class_centroids, n_classes, feature_dim, device):
    """
    Converts the class_centroids dictionary to a tensor for efficient distance calculation.
    Assumes class indices are 0..n_classes-1.
    """
    centroid_tensor = torch.zeros((n_classes, feature_dim), device=device)
    for class_idx, centroid in class_centroids.items():
        centroid_tensor[class_idx] = centroid.to(device)
    return centroid_tensor

# Global registry instance
expert_registry = ExpertRegistry()

def create_expert_profile(specialty_class_idx, archetype='high_specialist', expert_id=None, 
                         custom_params=None, dataset_name=None):
    """
    Create an expert profile with specified parameters.
    
    Args:
        specialty_class_idx: Class that the expert specialises in
        archetype: Predefined archetype or 'custom' for custom parameters
        expert_id: Unique identifier for the expert
        custom_params: Dict with custom alpha/beta values
        dataset_name: Name of the dataset (for unique naming)
    
    Returns:
        Expert profile dictionary
    """
    if expert_id is None:
        dataset_prefix = f"{dataset_name}_" if dataset_name else ""
        expert_id = f"expert_{dataset_prefix}{specialty_class_idx}_{archetype}"
    
    if archetype == 'custom' and custom_params is not None:
        alpha_spec = custom_params['alpha_specialty']
        beta_spec = custom_params['beta_specialty'] 
        alpha_non_spec = custom_params['alpha_non_specialty']
        beta_non_spec = custom_params['beta_non_specialty']
        archetype_name = 'Custom'
        description = 'Custom parameter configuration'
    elif archetype in EXPERT_ARCHETYPES:
        archetype_data = EXPERT_ARCHETYPES[archetype]
        alpha_spec, beta_spec = archetype_data['specialty_params']
        alpha_non_spec, beta_non_spec = archetype_data['non_specialty_params']
        archetype_name = archetype_data['name']
        description = archetype_data['description']
    else:
        raise ValueError(f"Unknown archetype: {archetype}. Use one of {list(EXPERT_ARCHETYPES.keys())} or 'custom'")
    
    expert_profile = {
        'expert_id': expert_id,
        'name': f'{archetype_name} (Class {specialty_class_idx})',
        'specialty_class': specialty_class_idx,
        'archetype': archetype,
        'description': description,
        'alpha_specialty': alpha_spec,
        'beta_specialty': beta_spec,
        'alpha_non_specialty': alpha_non_spec,
        'beta_non_specialty': beta_non_spec,
        'created_timestamp': pd.Timestamp.now().isoformat()
    }
    
    # Register the expert
    expert_registry.register_expert(expert_profile)
    
    return expert_profile

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
    
    # Normalise by maximum distance for this class
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

def simulate_prediction(config, image, true_label, expert_params, class_centroids, max_distances, trained_classifier, instance_index=None, centroid_tensor=None):
    """
    Simulate an expert's prediction for a single image.
    
    Args:
        config: Configuration object
        image: Input image tensor
        true_label: Ground truth label
        expert_params: Dictionary containing expert configuration
        class_centroids: Class centroids from simulation setup
        max_distances: Max distances for normalisation
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
        instance_features = features[0]

        # Compute prototypicality score
        proto_score = get_prototypicality(instance_features, true_label, class_centroids, max_distances)
        
        # Determine if this is the expert's specialty
        is_specialty = true_label == expert_params['specialty_class']
        
        # Get appropriate parameters
        if is_specialty:
            alpha = expert_params['alpha_specialty']
            beta = expert_params['beta_specialty']
        else:
            alpha = expert_params['alpha_non_specialty']
            beta = expert_params['beta_non_specialty']

        # Compute instance-specific accuracy
        accuracy = get_instance_accuracy(proto_score, alpha, beta)

        # Correlate correctness across experts using shared u_i per instance
        if instance_index is not None:
            base_seed = int(config.seed)
            instance_seed = (base_seed + int(instance_index)) % (2**32)
            rng_success = np.random.RandomState(instance_seed)
            u_i = rng_success.rand()
        else:
            u_i = np.random.random()
            instance_seed = None

        # Generate stochastic prediction based on shared u_i
        is_correct = u_i < accuracy
        
        if is_correct:
            prediction = true_label
        else:
            # Correlated error type: sample from softmax(-distance to centroids), masking true label
            if centroid_tensor is not None and config.n_classes > 1:
                distances = torch.norm(instance_features - centroid_tensor, dim=1)
                neg_distances = -distances
                masked_neg_distances = neg_distances.clone()
                masked_neg_distances[true_label] = float('-inf')
                error_distribution = F.softmax(masked_neg_distances, dim=0)
                # Deterministic sampling keyed on instance index (same for all experts on this instance)
                if instance_seed is not None:
                    # Mix in expert_id to avoid identical wrong-class across experts
                    expert_id_str = str(expert_params.get('expert_id', ''))
                    expert_hash = int.from_bytes(hashlib.md5(expert_id_str.encode()).digest()[:4], 'little')
                    error_seed = (instance_seed + 1000000 + expert_hash) % (2**32)
                    rng_error = np.random.RandomState(error_seed)
                else:
                    rng_error = np.random
                probs = error_distribution.detach().cpu().numpy()
                probs = probs / probs.sum()
                prediction = int(rng_error.choice(config.n_classes, p=probs))
                # In rare numeric cases, guard against sampling the true label
                if prediction == int(true_label):
                    possible_labels = [c for c in range(config.n_classes) if c != int(true_label)]
                    prediction = int(rng_error.choice(possible_labels))
            else:
                # Fallback: random incorrect label
                possible_labels = [i for i in range(config.n_classes) if i != true_label]
                if instance_seed is not None:
                    expert_id_str = str(expert_params.get('expert_id', ''))
                    expert_hash = int.from_bytes(hashlib.md5(expert_id_str.encode()).digest()[:4], 'little')
                    error_seed = (instance_seed + 1000000 + expert_hash) % (2**32)
                    rng_error = np.random.RandomState(error_seed)
                    prediction = int(rng_error.choice(possible_labels))
                else:
                    prediction = int(np.random.choice(possible_labels))
        
        # Convert all values to JSON-serializable types
        return {
            'true_label': int(true_label),  # Convert to native Python int
            'prediction': int(prediction),  # Convert numpy int to Python int
            'is_correct': bool(is_correct),  # Convert numpy bool to Python bool
            'proto_score': float(proto_score),  # Ensure it's a Python float
            'accuracy': float(accuracy),  # Ensure it's a Python float
            'is_specialty': bool(is_specialty),  # Convert to Python bool
            'alpha_used': float(alpha),  # Ensure it's a Python float
            'beta_used': float(beta)  # Ensure it's a Python float
        }

def check_expert_predictions_exist(expert_profile, datasets=['train', 'val', 'test'], save_dir='expert_predictions', config_dataset=None):
    """
    Check if predictions already exist for a specific expert profile.
    
    Args:
        expert_profile: Expert profile dictionary
        datasets: List of dataset names to check
        save_dir: Directory where predictions are saved
        config_dataset: Dataset name for subdirectory organization
    
    Returns:
        dict: {dataset_name: exists_bool}
    """
    existence_check = {}
    
    # Determine the actual save directory (with dataset subdirectory if specified)
    if config_dataset:
        actual_save_dir = os.path.join(save_dir, config_dataset)
    else:
        actual_save_dir = save_dir
    
    for dataset_name in datasets:
        filename = f"expert_{expert_profile['expert_id']}_{dataset_name}_predictions.json"
        filepath = os.path.join(actual_save_dir, filename)
        existence_check[dataset_name] = os.path.exists(filepath)
    
    return existence_check

def get_expert_params_signature(alpha_specialty, beta_specialty, alpha_non_specialty, beta_non_specialty):
    """
    Create a unique signature for expert parameters.
    
    Args:
        alpha_specialty: Alpha parameter for specialty class
        beta_specialty: Beta parameter for specialty class
        alpha_non_specialty: Alpha parameter for non-specialty classes
        beta_non_specialty: Beta parameter for non-specialty classes
    
    Returns:
        String signature for the parameter combination
    """
    return f"as{alpha_specialty:.6f}_bs{beta_specialty:.6f}_ans{alpha_non_specialty:.6f}_bns{beta_non_specialty:.6f}"

def check_predictions_exist_by_params(specialty_class, alpha_specialty, beta_specialty, 
                                     alpha_non_specialty, beta_non_specialty,
                                     datasets=['train', 'val', 'test'], save_dir='expert_predictions'):
    """
    Check if predictions exist for specific expert parameters.
    
    Args:
        specialty_class: Expert's specialty class
        alpha_specialty: Alpha parameter for specialty class
        beta_specialty: Beta parameter for specialty class  
        alpha_non_specialty: Alpha parameter for non-specialty classes
        beta_non_specialty: Beta parameter for non-specialty classes
        datasets: List of dataset names to check
        save_dir: Directory where predictions are saved
    
    Returns:
        dict: {dataset_name: exists_bool}
    """
    # Create a temporary expert profile to generate the expected filename
    temp_expert_id = f"expert_{specialty_class}_custom_{get_expert_params_signature(alpha_specialty, beta_specialty, alpha_non_specialty, beta_non_specialty)}"
    
    existence_check = {}
    
    for dataset_name in datasets:
        # Check multiple possible naming patterns
        possible_filenames = [
            f"expert_{temp_expert_id}_{dataset_name}_predictions.json",
            f"expert_{specialty_class}_custom_{dataset_name}_predictions.json",
            f"expert_{specialty_class}_high_specialist_{dataset_name}_predictions.json",  # Common archetype
        ]
        
        exists = False
        for filename in possible_filenames:
            filepath = os.path.join(save_dir, filename)
            if os.path.exists(filepath):
                # Double-check by reading the file and verifying parameters
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        expert_profile = data['metadata']['expert_profile']
                        
                        # Check if parameters match (with tolerance)
                        tolerance = 1e-6
                        param_match = (
                            abs(expert_profile['alpha_specialty'] - alpha_specialty) < tolerance and
                            abs(expert_profile['beta_specialty'] - beta_specialty) < tolerance and
                            abs(expert_profile['alpha_non_specialty'] - alpha_non_specialty) < tolerance and
                            abs(expert_profile['beta_non_specialty'] - beta_non_specialty) < tolerance and
                            expert_profile['specialty_class'] == specialty_class
                        )
                        
                        if param_match:
                            exists = True
                            break
                except (json.JSONDecodeError, KeyError):
                    continue
        
        existence_check[dataset_name] = exists
    
    return existence_check

def generate_expert_predictions_for_dataset(config, expert_profile, dataset, dataset_name, 
                                           class_centroids, max_distances, best_state, 
                                           batch_size=100, save_dir='expert_predictions'):
    """
    Generate predictions for an expert on a complete dataset and save as JSON.
    
    Args:
        config: Configuration object
        expert_profile: Expert profile dictionary
        dataset: PyTorch dataset
        dataset_name: Name of the dataset ('train', 'val', or 'test')
        class_centroids: Class centroids from prototypicality computation
        max_distances: Max distances for normalisation
        best_state: Best state dict from trained classifier
        batch_size: Batch size for processing
        save_dir: Directory to save JSON files
    
    Returns:
        Dictionary with all predictions and metadata
    """
    print(f"Generating predictions for Expert {expert_profile['expert_id']} on {dataset_name} data...")
    
    # Create dataset-specific save directory
    dataset_save_dir = os.path.join(save_dir, config.dataset)
    os.makedirs(dataset_save_dir, exist_ok=True)
    
    all_results = []
    num_samples = len(dataset)
    
    # Load the trained classifier once
    exp_clf = TrainableClassifier(config.n_classes, config.dataset)
    exp_clf.to(device)
    exp_clf.load_state_dict(best_state)
    exp_clf.eval()
    
    # Prepare centroid tensor once for correlated error sampling
    if class_centroids and config.n_classes > 0:
        first_key = next(iter(class_centroids))
        feature_dim = class_centroids[first_key].shape[0]
        centroid_tensor = centroids_to_tensor(class_centroids, config.n_classes, feature_dim, device)
    else:
        centroid_tensor = None

    # Process in batches for efficiency
    for i in tqdm(range(num_samples), 
                 desc=f"Expert {expert_profile['expert_id']} - {dataset_name}"):
        
        # Get image and label from dataset
        if hasattr(dataset, 'data') and hasattr(dataset, 'targets'):
            # Direct access for some datasets
            image = dataset.data[i]
            true_label = dataset.targets[i]
        else:
            # Use __getitem__ method
            image, true_label = dataset[i]
        
        # Convert to tensor if needed
        if not torch.is_tensor(image):
            image = torch.tensor(image)
        if not torch.is_tensor(true_label):
            true_label = torch.tensor(true_label)
            
        true_label_int = true_label.item() if torch.is_tensor(true_label) else int(true_label)
        
        # Generate prediction
        result = simulate_prediction(
            config, image, true_label_int, expert_profile,
            class_centroids, max_distances, exp_clf,
            instance_index=i, centroid_tensor=centroid_tensor
        )
        
        # Add sample index and expert info
        result['sample_idx'] = i
        result['expert_id'] = expert_profile['expert_id']
        result['expert_specialty_class'] = expert_profile['specialty_class']
        result['expert_archetype'] = expert_profile['archetype']
        result['dataset'] = dataset_name
        
        all_results.append(result)
    
    # Prepare data structure with metadata
    data_with_metadata = {
        'metadata': {
            'expert_profile': expert_profile,
            'dataset_name': dataset_name,
            'num_samples': len(all_results),
            'generation_timestamp': pd.Timestamp.now().isoformat(),
            'config_summary': {
                'dataset': config.dataset,
                'n_classes': config.n_classes,
                'seed': config.seed
            }
        },
        'predictions': all_results
    }
    
    # Save as JSON
    filename = f"expert_{expert_profile['expert_id']}_{dataset_name}_predictions.json"
    filepath = os.path.join(dataset_save_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(data_with_metadata, f, indent=2)
    
    accuracy = sum(r['is_correct'] for r in all_results) / len(all_results)
    print(f"Saved {len(all_results)} predictions to {filepath}")
    print(f"Expert {expert_profile['expert_id']} accuracy on {dataset_name}: {accuracy:.3f}")
    
    return data_with_metadata

def generate_or_load_expert_predictions(config, specialty_classes=None, expert_archetypes=None, 
                                       custom_experts=None, datasets=['train', 'val', 'test'],
                                       save_dir='expert_predictions', force_regenerate=False):
    """
    Generate expert predictions if they don't exist, or load existing ones.
    
    Args:
        config: Configuration object
        specialty_classes: List of specialty classes to create experts for (default: all classes)
        expert_archetypes: List of archetypes to create experts for (default: ['high_specialist'])
        custom_experts: List of custom expert specifications with parameters
        datasets: List of dataset names to generate predictions for
        save_dir: Directory to save/load predictions
        force_regenerate: If True, regenerate even if predictions exist
    
    Returns:
        List of prediction data dictionaries
    """
    print("="*60)
    print("EXPERT PREDICTIONS GENERATION/LOADING")
    print("="*60)
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Set defaults
    if specialty_classes is None:
        specialty_classes = list(range(config.n_classes))
    if expert_archetypes is None:
        expert_archetypes = ['high_specialist']
    if custom_experts is None:
        custom_experts = []
    
    # Load datasets
    print("Loading datasets...")
    dataset_objects = {}
    train_data, val_data, test_data, _ = load_dataset(config)
    if 'train' in datasets:
        dataset_objects['train'] = train_data
    if 'val' in datasets:
        dataset_objects['val'] = val_data
    if 'test' in datasets:
        dataset_objects['test'] = test_data
    
    # Load trained classifier and prototypicality data
    print("Loading trained classifier and prototypicality data...")
    from train_expert_clf import train_and_compute_prototypicality
    best_state, class_centroids, max_distances = train_and_compute_prototypicality(config)
    
    # Create expert profiles
    expert_profiles = []
    
    # Standard archetype experts
    for class_idx in specialty_classes:
        for archetype in expert_archetypes:
            expert_profile = create_expert_profile(class_idx, archetype, dataset_name=config.dataset)
            expert_profiles.append(expert_profile)
    
    # Custom experts
    for custom_expert in custom_experts:
        expert_profile = create_expert_profile(
            specialty_class_idx=custom_expert['specialty_class'],
            archetype='custom',
            expert_id=custom_expert.get('expert_id'),
            custom_params=custom_expert['params'],
            dataset_name=config.dataset
        )
        expert_profiles.append(expert_profile)
    
    print(f"Created {len(expert_profiles)} expert profiles")
    
    # Check which predictions already exist and generate missing ones
    all_predictions = []
    
    for expert_profile in expert_profiles:
        expert_id = expert_profile['expert_id']
        print(f"\n--- Checking Expert {expert_id} ---")
        print(f"    Speciality: Class {expert_profile['specialty_class']}")
        print(f"    Archetype: {expert_profile['archetype']}")
        print(f"    Specialty α={expert_profile['alpha_specialty']:.3f}, β={expert_profile['beta_specialty']:.3f}")
        print(f"    Non-specialty α={expert_profile['alpha_non_specialty']:.3f}, β={expert_profile['beta_non_specialty']:.3f}")
        
        # Check which datasets already have predictions
        existing_predictions = check_expert_predictions_exist(expert_profile, datasets, save_dir, config.dataset)
        
        for dataset_name in datasets:
            if force_regenerate or not existing_predictions[dataset_name]:
                if existing_predictions[dataset_name]:
                    print(f"  Force regenerating {dataset_name} predictions...")
                else:
                    print(f"  Generating {dataset_name} predictions...")
                
                result = generate_expert_predictions_for_dataset(
                    config=config,
                    expert_profile=expert_profile,
                    dataset=dataset_objects[dataset_name],
                    dataset_name=dataset_name,
                    class_centroids=class_centroids,
                    max_distances=max_distances,
                    best_state=best_state,
                    save_dir=save_dir
                )
                all_predictions.append(result)
            else:
                print(f"  ✓ {dataset_name} predictions already exist, loading...")
                filename = f"expert_{expert_profile['expert_id']}_{dataset_name}_predictions.json"
                dataset_save_dir = os.path.join(save_dir, config.dataset)
                filepath = os.path.join(dataset_save_dir, filename)
                with open(filepath, 'r') as f:
                    result = json.load(f)
                    all_predictions.append(result)
    
    # Save/update expert registry
    registry_data = {
        'expert_registry': expert_registry.list_all_experts(),
        'archetype_definitions': EXPERT_ARCHETYPES,
        'generation_summary': {
            'total_experts': len(expert_profiles),
            'datasets': datasets,
            'total_prediction_files': len(all_predictions),
            'last_updated': pd.Timestamp.now().isoformat()
        }
    }
    
    # Save registry in dataset-specific directory
    dataset_save_dir = os.path.join(save_dir, config.dataset)
    registry_filepath = os.path.join(dataset_save_dir, 'expert_registry.json')
    with open(registry_filepath, 'w') as f:
        json.dump(registry_data, f, indent=2)
    
    # Compute and save error-correlation summaries
    corr_summary, corr_path = save_error_correlation_summary(
        all_predictions=all_predictions,
        datasets=datasets,
        n_classes=int(config.n_classes),
        save_dir=save_dir,
        config_dataset=config.dataset
    )

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Number of experts processed: {len(expert_profiles)}")
    print(f"Datasets: {datasets}")
    print(f"Total prediction files: {len(all_predictions)}")
    print(f"Files location: {save_dir}")
    print(f"Expert registry updated: {registry_filepath}")
    # Brief correlation summary
    for ds in datasets:
        ds_sum = corr_summary['datasets'].get(ds, {})
        avg_corr = ds_sum.get('avg_pairwise_correlation')
        n_exp = ds_sum.get('num_experts', 0)
        n_samp = ds_sum.get('num_samples', 0)
        print(f"Correlation [{ds}]: experts={n_exp}, samples={n_samp}, avg_pairwise_corr={avg_corr}")
    print(f"Correlation summary saved: {corr_path}")
    
    return all_predictions

def load_expert_predictions(save_dir='expert_predictions', expert_id=None, dataset=None, config_dataset=None):
    """
    Load expert predictions from saved JSON files.
    
    Args:
        save_dir: Directory where JSON files are saved
        expert_id: Specific expert ID to load (None for all)
        dataset: Specific dataset to load ('train', 'val', 'test', or None for all)
        config_dataset: Dataset name for subdirectory organization
    
    Returns:
        List of prediction data dictionaries
    """
    all_data = []
    
    # Determine the actual directory to search
    if config_dataset:
        search_dir = os.path.join(save_dir, config_dataset)
    else:
        search_dir = save_dir
    
    if not os.path.exists(search_dir):
        print(f"Warning: Directory {search_dir} does not exist")
        return all_data
    
    for filename in os.listdir(search_dir):
        if filename.endswith('.json') and 'predictions' in filename and filename != 'expert_registry.json' and filename != 'predictions_summary.json':
            # Filter by expert_id if specified
            if expert_id and expert_id not in filename:
                continue
            # Filter by dataset if specified  
            if dataset and dataset not in filename:
                continue
                
            filepath = os.path.join(search_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                all_data.append(data)
    
    return all_data

# =====================
# Correlation analytics
# =====================

def _align_predictions_by_dataset(all_predictions, target_dataset):
    """
    Align predictions across experts for a given dataset by sample_idx.

    Returns:
        expert_ids: List[str]
        error_matrix: np.ndarray of shape (num_experts, num_samples), values {0,1}
        pred_matrix: np.ndarray of shape (num_experts, num_samples), dtype int
        true_labels: np.ndarray of shape (num_samples,), dtype int
    """
    # Group by expert and collect predictions for the target dataset
    expert_to_preds = {}
    for data in all_predictions:
        md = data.get('metadata', {})
        if md.get('dataset_name') != target_dataset:
            continue
        preds = data.get('predictions', [])
        if not preds:
            continue
        expert_id = md['expert_profile']['expert_id']
        expert_to_preds[expert_id] = preds

    if not expert_to_preds:
        return [], np.zeros((0, 0), dtype=int), np.zeros((0, 0), dtype=int), np.zeros((0,), dtype=int)

    # Determine num_samples via union of sample_idx across experts
    all_indices = set()
    for preds in expert_to_preds.values():
        for p in preds:
            all_indices.add(int(p['sample_idx']))
    num_samples = max(all_indices) + 1 if all_indices else 0

    expert_ids = sorted(expert_to_preds.keys())
    num_experts = len(expert_ids)
    error_matrix = np.zeros((num_experts, num_samples), dtype=int)
    pred_matrix = np.full((num_experts, num_samples), fill_value=-1, dtype=int)
    true_labels = np.full((num_samples,), fill_value=-1, dtype=int)

    for e_idx, eid in enumerate(expert_ids):
        for p in expert_to_preds[eid]:
            sidx = int(p['sample_idx'])
            error_matrix[e_idx, sidx] = 0 if bool(p['is_correct']) else 1
            pred_matrix[e_idx, sidx] = int(p['prediction'])
            if true_labels[sidx] == -1:
                true_labels[sidx] = int(p['true_label'])

    # In rare cases if any true label remains -1 (missing across all experts), set to 0
    if (true_labels == -1).any():
        true_labels[true_labels == -1] = 0

    return expert_ids, error_matrix, pred_matrix, true_labels


def _compute_pairwise_error_stats(error_matrix):
    """
    Compute pairwise binary-error correlations and related stats.

    Returns dict with matrices as python lists.
    """
    num_experts = error_matrix.shape[0]
    if num_experts == 0:
        return {
            'pairwise_error_correlation': [],
            'pairwise_coerror_rate': [],
            'pairwise_jaccard_error': [],
            'avg_pairwise_correlation': None,
        }

    N = max(1, error_matrix.shape[1])
    corr = np.full((num_experts, num_experts), np.nan, dtype=float)
    coerr = np.zeros((num_experts, num_experts), dtype=float)
    jacc = np.zeros((num_experts, num_experts), dtype=float)

    # Precompute means and variances for each expert
    means = error_matrix.mean(axis=1)
    vars_ = means * (1.0 - means)

    for i in range(num_experts):
        for j in range(num_experts):
            Xi = error_matrix[i]
            Xj = error_matrix[j]
            inter = float(np.mean((Xi == 1) & (Xj == 1)))
            coerr[i, j] = inter
            # Jaccard over error sets
            union = float(np.mean((Xi == 1) | (Xj == 1)))
            jacc[i, j] = (inter / union) if union > 0 else 0.0
            # Pearson correlation (phi) for 0/1
            denom = np.sqrt(vars_[i] * vars_[j])
            if denom > 0:
                corr[i, j] = (float(np.mean(Xi * Xj)) - means[i] * means[j]) / denom
            else:
                corr[i, j] = np.nan

    # Average off-diagonal, ignoring NaNs
    if num_experts > 1:
        mask = ~np.eye(num_experts, dtype=bool)
        avg_corr = float(np.nanmean(corr[mask])) if np.any(mask) else None
    else:
        avg_corr = None

    return {
        'pairwise_error_correlation': corr.tolist(),
        'pairwise_coerror_rate': coerr.tolist(),
        'pairwise_jaccard_error': jacc.tolist(),
        'avg_pairwise_correlation': avg_corr,
    }


def _compute_wrong_confusion(true_labels, pred_matrix, n_classes):
    """
    Aggregate a confusion-style matrix over all experts counting only wrong predictions.

    Returns n_classes x n_classes counts, where [t, p] counts wrong predictions of class t as p.
    """
    conf = np.zeros((n_classes, n_classes), dtype=int)
    if pred_matrix.size == 0:
        return conf.tolist()

    num_experts = pred_matrix.shape[0]
    for e in range(num_experts):
        preds = pred_matrix[e]
        for idx, p in enumerate(preds):
            if p < 0:
                continue
            t = int(true_labels[idx])
            if p != t:
                if 0 <= t < n_classes and 0 <= p < n_classes:
                    conf[t, p] += 1
    return conf.tolist()


def save_error_correlation_summary(all_predictions, datasets, n_classes, save_dir, config_dataset):
    """
    Compute and persist error-correlation summaries per dataset.
    """
    summary = {
        'datasets': {},
        'meta': {
            'generation_timestamp': pd.Timestamp.now().isoformat()
        }
    }

    for ds in datasets:
        expert_ids, error_matrix, pred_matrix, true_labels = _align_predictions_by_dataset(all_predictions, ds)
        stats = _compute_pairwise_error_stats(error_matrix)
        wrong_conf = _compute_wrong_confusion(true_labels, pred_matrix, n_classes)
        summary['datasets'][ds] = {
            'expert_ids': expert_ids,
            'num_experts': len(expert_ids),
            'num_samples': int(error_matrix.shape[1] if error_matrix.size else 0),
            'avg_pairwise_correlation': stats['avg_pairwise_correlation'],
            'pairwise_error_correlation': stats['pairwise_error_correlation'],
            'pairwise_coerror_rate': stats['pairwise_coerror_rate'],
            'pairwise_jaccard_error': stats['pairwise_jaccard_error'],
            'wrong_confusion_overall': wrong_conf,
        }

    dataset_save_dir = os.path.join(save_dir, config_dataset)
    os.makedirs(dataset_save_dir, exist_ok=True)
    out_path = os.path.join(dataset_save_dir, 'predictions_summary.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    return summary, out_path

# Main interface functions for external use
def ensure_expert_predictions_exist(config, specialty_classes=None, expert_archetypes=['high_specialist'],
                                   custom_experts=None, datasets=['train', 'val', 'test'], 
                                   save_dir='expert_predictions'):
    """
    Ensure expert predictions exist for specified configuration. Generate if missing.
    
    This is the main function to call from other scripts.
    
    Args:
        config: Configuration object
        specialty_classes: List of specialty classes (default: all classes)
        expert_archetypes: List of expert archetypes to create
        custom_experts: List of custom expert configurations
        datasets: List of datasets to ensure predictions for
        save_dir: Directory for saving/loading predictions
    
    Returns:
        List of prediction data dictionaries
    """
    return generate_or_load_expert_predictions(
        config=config,
        specialty_classes=specialty_classes,
        expert_archetypes=expert_archetypes,
        custom_experts=custom_experts,
        datasets=datasets,
        save_dir=save_dir,
        force_regenerate=False
    )

if __name__ == "__main__":
    # Example usage
    from attrdict import AttrDict
    
    # Create a sample config for organs_axial
    config = AttrDict()
    config.dataset = 'organs_axial'
    config.n_classes = 11
    config.seed = 42
    
    # Generate expert predictions with multiple archetypes
    predictions = ensure_expert_predictions_exist(
        config, 
        expert_archetypes=['variable_specialist', 'realistic_specialist']
    )
