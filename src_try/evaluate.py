import sys
import os
import json
import torch
import numpy as np
import pandas as pd
from attrdict import AttrDict
from torch.utils.data import DataLoader
from helper_fncs import load_dataset, load_model, bayesian_inference_per_human
from context_data import get_context_data, ContextSampler
from experts import simulate_experts
from datasets import SyntheticHumanPredictionDataset
from utils import set_seed, ROOT
import tqdm

# Add directory to path so we can import from other files
ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(ROOT)


device = 'cuda'

def list_options(directory):
    """List available subdirectories in the given directory."""
    if not os.path.exists(directory):
        return []
    return [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]

def prompt_with_hints(prompt_message, options):
    """Prompt user with available options and return their selection by index."""
    if options:
        print(f"\nAvailable options for {prompt_message}:")
        for i, option in enumerate(options):
            print(f"  [{i}] {option}")
    else:
        print(f"\nNo options found for {prompt_message}. Please check your directory structure.")
        return None
    
    while True:
        try:
            selection = int(input(f"Select the index for {prompt_message}: ").strip())
            if 0 <= selection < len(options):
                return options[selection]
            else:
                print(f"Invalid index. Please select a number between 0 and {len(options) - 1}.")
        except ValueError:
            print("Invalid input. Please enter a valid index.")

def get_experiment_paths(root, experiment_name, model, dataset, p_out):
    base_dir = os.path.join(root, "experiments", experiment_name, model, dataset, p_out)
    if not os.path.exists(base_dir):
        print(f"No experiments found at: {base_dir}")
        return []

    paths = []
    for seed_dir in os.listdir(base_dir):
        seed_path = os.path.join(base_dir, seed_dir)
        if os.path.isdir(seed_path):
            for timestamp_dir in os.listdir(seed_path):
                full_path = os.path.join(seed_path, timestamp_dir)
                if os.path.isdir(full_path):
                    paths.append(full_path)
    return paths

def compute_partial_auc(accuracy_array, start_frac=0, end_frac=1, normalise=False):
    """
    Computes the area under a portion of the curve (AUC) using the trapezoidal rule.

    Parameters:
        accuracy_array (array-like): An array of accuracy values.
        start_frac (float): The starting fraction of x-values (0.0 to 1.0).
        end_frac (float): The ending fraction of x-values (0.0 to 1.0).
        normalise (bool): Whether to normalise the computed AUC.

    Returns:
        float: The computed partial area under the curve.
    """
    # Ensure the input is a numpy array
    accuracy_array = np.array(accuracy_array)

    # Calculate the range of indices for the selected fraction
    total_points = len(accuracy_array)
    start_idx = int(start_frac * total_points)
    end_idx = int(end_frac * total_points)

    # Subset the array for the desired portion
    subset_array = accuracy_array[start_idx:end_idx]

    # Compute the AUC using the trapezoidal rule
    auc = np.trapz(subset_array, dx=1)

    if normalise:
        auc = auc / (end_idx - start_idx)

    return auc

def gen_predictions(config, model, experts_lst, data_loader, cntx_sampler):
    n_classes = config['n_classes']
    model.eval()
    with torch.no_grad():
        all_outputs = []
        for batch in data_loader:
            query_x_batch = batch['xc'][:, 0, :, :, :]
            query_x_batch = query_x_batch.to(device)
            expert_cntx = cntx_sampler.cntx

            if config['model'] == 'ifd':
                # Include variance information if needed
                m, weighted_representations, variances = bayesian_inference_per_human(
                    expert_cntx.mc, expert_cntx.yc[0], n_classes, return_variance=True
                )
                outputs = model(query_x_batch, weighted_representations, variances)
            elif config['model'] == 'pop-avg':
                outputs = model(query_x_batch).unsqueeze(0).repeat(len(experts_lst), 1, 1)
            elif config['model'] == 'l2d-pop':
                outputs = model(query_x_batch, expert_cntx)
            elif config['model'] == 'l2d-multi':
                outputs = model(query_x_batch) 
                outputs = outputs.unsqueeze(0).repeat(config["n_experts"],1,1)

            all_outputs.append(outputs)
    all_outputs_stacked = torch.cat(all_outputs, dim=1)
    return all_outputs_stacked

def evaluate(config, outputs, data_loader, experts_classes, start_frac, end_frac):
    n_classes = config['n_classes']
    m_outputs = outputs[:, :, :]
    df_lst = []

    for dim in range(m_outputs.size()[0]):
        clf_preds = m_outputs[dim][:, :n_classes].argmax(dim=-1).cpu()
        def_probs = m_outputs[dim][:, -1].cpu()
        def_conf = (m_outputs[dim][:, :n_classes].max(dim=-1)[0] - m_outputs[dim][:, -1]).cpu()
        def_flag = (m_outputs[dim][:, :n_classes].max(dim=-1)[0] < m_outputs[dim][:, -1]).int().cpu()
        df = pd.DataFrame({
            'clf_preds': clf_preds,
            'def_probs': def_probs,
            'def_conf': def_conf,
            'def_flag': def_flag,
            'labels': data_loader.dataset.yc[0],
            'exp_preds': data_loader.dataset.mc[dim],
            'oracle': experts_classes[dim]
        })
        df_lst.append(df)

    df_a = pd.concat([df.reset_index() for df in df_lst])
    df_a.reset_index(drop=True, inplace=True)
    df_a.sort_values(by='def_conf', inplace=True)
    df_a = df_a.drop_duplicates(subset=['index'], keep='first')

    sys_acc_arr = [
        ((df_a.iloc[:i+1].exp_preds == df_a.iloc[:i+1].labels).sum() +
         (df_a.iloc[i+1:].clf_preds == df_a.iloc[i+1:].labels).sum()) / len(df_a)
        for i in range(len(df_a))
    ]

    def_acc_arr = [
        (df_a.iloc[:i+1].exp_preds == df_a.iloc[:i+1].labels).mean()
        for i in range(len(df_a))
    ]

    ausac = compute_partial_auc(sys_acc_arr, start_frac=start_frac, end_frac=end_frac, normalise=True) 
    audac = compute_partial_auc(def_acc_arr, start_frac=start_frac, end_frac=end_frac, normalise=True)
    return ausac, audac

def gogogo(path, start_frac, end_frac):
    config_path = os.path.join(path, 'config.json')
    with open(config_path, 'r') as config_file:
        config_dict = json.load(config_file)
    config = AttrDict(config_dict)

    set_seed(config["seed"])
    train_data, val_data, test_data, _ = load_dataset(config)
    experts_id, experts_ood = simulate_experts(config)
    ood_experts_expertise_classes = [i.classes_oracle[0] for i in experts_ood]
    id_experts_expertise_classes = [i.classes_oracle[0] for i in experts_id]

    X_c, y_c, X_tr_q, y_tr_q, X_tst_q, y_tst_q, (cntx_indices, query_indices) = get_context_data(config, train_data, val_data, test_data)

    from generate_exp_preds import ensure_expert_predictions_exist
    expert_predictions = ensure_expert_predictions_exist(
        config=config,
        specialty_classes=list(range(config.n_classes)),  # All classes
        expert_archetypes=config['expert_archetypes'],
        datasets=['train', 'val', 'test'],
        save_dir='expert_predictions'
    )

    from main import extract_expert_predictions_for_sampled_experts, extract_expert_predictions_for_validation

    m_c_id, _ = extract_expert_predictions_for_sampled_experts(
        expert_predictions=expert_predictions,
        experts_expertise_classes=id_experts_expertise_classes,
        cntx_indices=cntx_indices,
        query_indices=query_indices,
        y_c=y_c,
        y_tr_q=y_tr_q,
        dataset_name='train',
        verbose=True
    )
    m_c_ood, _ = extract_expert_predictions_for_sampled_experts(
        expert_predictions=expert_predictions,
        experts_expertise_classes=ood_experts_expertise_classes,
        cntx_indices=cntx_indices,
        query_indices=query_indices,
        y_c=y_c,
        y_tr_q=y_tr_q,
        dataset_name='train',
        verbose=True
    )
    m_te_id = extract_expert_predictions_for_validation(
        expert_predictions=expert_predictions,
        experts_expertise_classes=id_experts_expertise_classes,  # Same experts as training
        y_val=y_tst_q,  # or whatever your validation labels are called
        dataset_name='test',
        verbose=True
    )
    m_te_ood = extract_expert_predictions_for_validation(
        expert_predictions=expert_predictions,
        experts_expertise_classes=ood_experts_expertise_classes,  # Same experts as training
        y_val=y_tst_q,  # or whatever your validation labels are called
        dataset_name='test',
        verbose=True
    )


    kwargs = {'num_workers': 0, 'pin_memory': True}
    cntx_sampler_test_id = ContextSampler(config, X_c, y_c, X_tst_q, y_tst_q, experts_id, config["n_cntx_pts"], context_expert_preds=m_c_id, query_expert_preds=m_te_id, device=device, **kwargs)
    cntx_sampler_test_ood = ContextSampler(config, X_c, y_c, X_tst_q, y_tst_q, experts_ood, config["n_cntx_pts"], context_expert_preds=m_c_ood, query_expert_preds=m_te_ood, device=device, **kwargs)
    cntx_sampler_test_id.send_context_to_device()
    cntx_sampler_test_ood.send_context_to_device()

    n_experts = len(experts_id)
    config["n_experts"] = n_experts

    best_statedict = torch.load(os.path.join(path, 'best_val_acc_sd.pth'), weights_only=True)
    model = load_model(config, best_statedict)

    tst_query_dataset_id = SyntheticHumanPredictionDataset(cntx_sampler_test_id.query)
    tst_query_loader_id = DataLoader(tst_query_dataset_id, batch_size=10000, shuffle=False)
    outputs_id = gen_predictions(config, model, experts_id, tst_query_loader_id, cntx_sampler_test_id)
    experts_classes_id = [i.classes_oracle[0] for i in experts_id]
    ausac_id, audac_id = evaluate(config, outputs_id, tst_query_loader_id, experts_classes_id, start_frac, end_frac)

    if config['model'] != 'l2d-multi':
        tst_query_dataset_ood = SyntheticHumanPredictionDataset(cntx_sampler_test_ood.query)
        tst_query_loader_ood = DataLoader(tst_query_dataset_ood, batch_size=10000, shuffle=False)
        outputs_ood = gen_predictions(config, model, experts_ood, tst_query_loader_ood, cntx_sampler_test_ood)
        experts_classes_ood = [i.classes_oracle[0] for i in experts_ood]
        ausac_ood, audac_ood = evaluate(config, outputs_ood, tst_query_loader_ood, experts_classes_ood, start_frac, end_frac)
        return ausac_id, audac_id, ausac_ood, audac_ood
    
    return ausac_id, audac_id

if __name__ == "__main__":
    ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # Get command-line inputs for start_frac and end_frac
    try:
        start_frac = float(input("Enter the start fraction (0 to 1): ").strip())
        end_frac = float(input("Enter the end fraction (0 to 1): ").strip())
        if not (0 <= start_frac <= 1 and 0 <= end_frac <= 1 and start_frac < end_frac):
            raise ValueError("Fractions must be in [0, 1] and start_frac < end_frac.")
    except ValueError as e:
        sys.exit(f"Invalid input for fractions: {e}")

    # Remaining options unchanged
    experiment_names = list_options(os.path.join(ROOT, "experiments"))
    experiment_name = prompt_with_hints("experiment name", experiment_names)
    if not experiment_name:
        sys.exit("No valid experiment name provided. Exiting.")

    model_options = list_options(os.path.join(ROOT, "experiments", experiment_name))
    model = prompt_with_hints("model", model_options)
    if not model:
        sys.exit("No valid model provided. Exiting.")

    dataset_options = list_options(os.path.join(ROOT, "experiments", experiment_name, model))
    dataset = prompt_with_hints("dataset", dataset_options)
    if not dataset:
        sys.exit("No valid dataset provided. Exiting.")

    p_out_options = list_options(os.path.join(ROOT, "experiments", experiment_name, model, dataset))
    p_out = prompt_with_hints("p_out", p_out_options)
    if not p_out:
        sys.exit("No valid p_out provided. Exiting.")

    experiment_paths = get_experiment_paths(ROOT, experiment_name, model, dataset, p_out)

    if not experiment_paths:
        sys.exit("No experiments found. Exiting.")

    print("\nFound the following experiments:")
    for i, path in enumerate(experiment_paths):
        print(f"[{i}] {path}")

    use_all = input("\nUse all experiments? (y/n): ").strip().lower() == 'y'
    if not use_all:
        selected_indices = input("Enter the indices of experiments to use (comma-separated): ").strip()
        selected_indices = [int(i) for i in selected_indices.split(",")]
        experiment_paths = [experiment_paths[i] for i in selected_indices]

    print("\n\nRunning...")
    results = [gogogo(path, start_frac, end_frac) for path in tqdm.tqdm(experiment_paths)]

    ausacs_id = [res[0] for res in results]
    audacs_id = [res[1] for res in results]

    print(f"\nAggregate Results (n={len(experiment_paths)}):")
    print(f"Settings: Experiment Name='{experiment_name}', Model='{model}', Dataset='{dataset}', p_out='{p_out}', Start Fraction={start_frac}, End Fraction={end_frac}")

    print("Average AUSAC ID: " + f"{np.mean(ausacs_id):.2f}".lstrip('0') + " ± " + f"{np.std(ausacs_id):.2f}".lstrip('0'))
    print("Average AUDAC ID: " + f"{np.mean(audacs_id):.2f}".lstrip('0') + " ± " + f"{np.std(audacs_id):.2f}".lstrip('0'))

    if model != 'l2d-multi':
        ausacs_ood = [res[2] for res in results]
        audacs_ood = [res[3] for res in results]
        print("Average AUSAC OOD: " + f"{np.mean(ausacs_ood):.2f}".lstrip('0') + " ± " + f"{np.std(ausacs_ood):.2f}".lstrip('0'))
        print("Average AUDAC OOD: " + f"{np.mean(audacs_ood):.2f}".lstrip('0') + " ± " + f"{np.std(audacs_ood):.2f}".lstrip('0'))