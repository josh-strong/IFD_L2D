import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import argparse
import os
import json
from datetime import datetime 
import pandas as pd
from copy import deepcopy
from matplotlib import pyplot as plt
import time
from tqdm import tqdm

# Local imports
from experts import simulate_experts
from context_data import get_context_data, ContextSampler
from datasets import SyntheticHumanPredictionDataset
from helper_fncs import validate, load_dataset
from attrdict import AttrDict
from utils import set_seed
from helper_fncs import load_model, bayesian_inference_per_human, cross_entropy_mod, cross_entropy_l2dmultiexp, random_mode_per_row, compute_auc
from train_expert_clf import train_and_compute_prototypicality

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Robust string-to-bool converter for argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("true", "1", "yes", "y", "t", "on")

# Define the command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Run model training with specified parameters.")
    parser.add_argument("--model", type=str, required=True, choices=["ifd", "l2d-pop", "l2d-multi", "pop-avg"], help="Which model to use")
    parser.add_argument("--with_attn", type=str2bool, default=True, help="Whether to use attention")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--dataset", type=str, required=True, choices=['ham10000','bus','organs_axial','blood_mnist', 'oct', 'new_ham10000', 'cifar10', 'imagenet16_grey'], help="Dataset")
    parser.add_argument("--lr_wrn", type=float, default=.008, help="Learning rate for WRN")
    parser.add_argument('--weight_decay', type=float, default=5e-05, help='Learning rate for weight decay.' )
    parser.add_argument("--train_batch_size", type=int, default=128, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=128, help="Validation batch size")
    parser.add_argument("--n_cntx_pts", type=int, default=50, help="Number of context points")
    parser.add_argument("--n_ID_experts", type=int, help="Number of in-distribution experts")
    parser.add_argument("--p_out", type=float, default=0.2, help="p_out value for experts")
    parser.add_argument("--p_in", type=float, default=1.0, help="p_in value for experts")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--early_stopping", type=int, default=250, help="Early stopping patience")
    parser.add_argument("--experiment_name", type=str, default="", help="Name of the experiment")
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with additional output')
    parser.add_argument('--lcb', action='store_true', default=False, help='Enable LCb')
    parser.add_argument('--lcb_alpha', type=float, default=1, help='LCb alpha')
    parser.add_argument('--ablation_weights', action='store_true', default=False, help='Ablate weights')
    parser.add_argument("--mode", type=str, default="ID", choices=["ID", "OOD"], help="Mode to run the script in (ID or OOD)")
    parser.add_argument(
        "--expert_archetypes",
        type=str,
        default="high_specialist",
        help="Comma-separated list of expert archetypes to use (e.g., 'high_specialist,moderate_specialist')"
    )
    parser.add_argument('--aux_loss', action='store_true', default=False, help='Enable auxiliary classification loss for IFD')
    parser.add_argument('--aux_loss_lambda', type=float, default=1.0, help='Balancing hyperparameter for auxiliary loss')
    return parser.parse_args()

# Set the config using the parsed arguments
def set_config(args):
    config = AttrDict()
    config["with_softmax"] = True
    config["with_attn"] = args.with_attn
    config["seed"] = args.seed
    config["lr_wrn"] = args.lr_wrn
    config["train_batch_size"] = args.train_batch_size
    config["val_batch_size"] = args.val_batch_size
    config["n_cntx_pts"] = args.n_cntx_pts
    config["p_out"] = args.p_out
    config["p_in"] = args.p_in  
    config['model'] = args.model
    config["decouple"] = False
    config['norm_type'] = 'batchnorm'
    config["dataset"] = args.dataset
    config['num_epochs'] = args.num_epochs
    config['early_stopping'] = args.early_stopping
    config['weight_decay'] = args.weight_decay
    config['experiment_name'] = args.experiment_name
    config['debug'] = args.debug
    config['lcb'] = args.lcb
    config['lcb_alpha'] = args.lcb_alpha
    config['ablation_weights'] = args.ablation_weights
    config['expert_archetypes'] = [a.strip() for a in args.expert_archetypes.split(',')]
    config['aux_loss'] = args.aux_loss
    config['aux_loss_lambda'] = args.aux_loss_lambda

    config['depth_embed'] = 6
    config['depth_reject'] = 4
    config['dim_hid'] = 256

    
    config['l2d'] = 'pop'
    config['setup'] = 'weighted m_approx'
    config['pretrain_lr'] = 1e-4
    config['pretrain_epochs'] = 100
    config['pretrain_early_stopping'] = 25

    # Set number of classes based on dataset
    if config['dataset'] == 'ham10000':
        config['n_classes'] = 7
        config['lr_wrn'] = 0.0001
    elif config['dataset'] == 'new_ham10000':
        config['n_classes'] = 7
        config['lr_wrn'] = 0.0001
    elif config['dataset'] == 'bus':
        config['n_classes'] = 3
        config['lr_wrn'] = 0.001
    elif config['dataset'] == 'organs_axial':
        config['n_classes'] = 11
        config['lr_wrn'] = 0.001
    elif config['dataset'] == 'blood_mnist':
        config['n_classes'] = 8
        config['lr_wrn'] = 0.00001
        config['pretrain_lr'] = 1e-4
        config['pretrain_epochs'] = 100
        config['pretrain_early_stopping'] = 25
        config['train_batch_size'] = 128
        config['val_batch_size'] = 128
        config['num_epochs'] = 250
        config['early_stopping'] = 50
    elif config['dataset'] == 'oct':
        config['n_classes'] = 4
        config['lr_wrn'] = 0.0001
        config['pretrain_lr'] = 1e-3
        config['pretrain_epochs'] = 100
        config['pretrain_early_stopping'] = 25
        config['train_batch_size'] = 128
        config['val_batch_size'] = 128
        config['num_epochs'] = 250
        config['early_stopping'] = 50
    elif config['dataset'] == 'cifar10':
        config['n_classes'] = 10
        config['lr_wrn'] = 0.00001
        config['train_batch_size'] = 64
        config['val_batch_size'] = 64
        config['n_cntx_pts'] = 25  
    elif config['dataset'] == 'imagenet16_grey':
        config['n_classes'] = 16
        config['lr_wrn'] = 0.001
        config['train_batch_size'] = 128
        config['val_batch_size'] = 128
        config['n_cntx_pts'] = 35
    else:
        raise ValueError("Dataset not supported")
    return config

import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, base_model, fc):
        super(BaseModel, self).__init__()
        self.base_model = base_model
        self.fc = fc
    
    def forward(self, x):
        x = self.base_model(x)
        x = self.fc(x)
        return x

def pretrain_clf(model, train_loader, val_loader, config):
    """Pretrain the classifier component of the model."""
    print("\nPretraining classifier...")
    base_model = BaseModel(model.classifier, model.fc)
    base_model = base_model.to(device)

    optimizer = torch.optim.Adam(base_model.parameters(), lr=config['pretrain_lr'])
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_val_acc = 0
    patience_counter = 0

    # Main training loop
    for epoch in range(config['pretrain_epochs']):
        # Training phase
        base_model.train()
        train_loss = 0
        train_acc = 0
        
        for batch in train_loader:
            query_x_batch = batch['xc'][:, 0, :, :, :].to(device)
            query_label_batch = batch['yc'][:, 0].to(device)

            optimizer.zero_grad()
            outputs = base_model(query_x_batch)
            loss = loss_fn(outputs, query_label_batch)
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            acc = (outputs.argmax(dim=1) == query_label_batch).float().mean()
            train_loss += loss.item()
            train_acc += acc.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)

        # Validation phase
        base_model.eval()
        val_loss = 0
        val_acc = 0

        with torch.no_grad():
            for batch in val_loader:
                query_x_batch = batch['xc'][:, 0, :, :, :].to(device)
                query_label_batch = batch['yc'][:, 0].to(device)

                outputs = base_model(query_x_batch)
                loss = loss_fn(outputs, query_label_batch)
                acc = (outputs.argmax(dim=1) == query_label_batch).float().mean()
                
                val_loss += loss.item()
                val_acc += acc.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)

        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f}")

        # Early stopping check
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            patience_counter = 0
            best_val_statedict = deepcopy(model)
            print(f"New best validation accuracy: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config['pretrain_early_stopping']:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"Best validation accuracy: {best_val_acc:.4f}")
                break

    return best_val_statedict.classifier.state_dict(), best_val_statedict.fc.state_dict()


def train_model(experts_lst, tr_cntx_sampler, val_cntx_sampler, config, model_statedict=None, num_epochs=5, plot_losses=False, early_stopping=None, return_val_loss=False, pretrained_clf_statedict=None):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = f"./experiments/{config['experiment_name']}/{config['model']}/{config['dataset']}/{config['expert_archetypes']}/{config['seed']}/{timestamp}"
    
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

    n_classes = config["n_classes"]
    n_experts = len(experts_lst)
    config["n_experts"] = n_experts
    
    # Save the config file to the directory
    config_dict = dict(config)  # Convert AttrDict to a regular dictionary
    # Ensure lists are serializable
    if 'classes_oracle_id' in config_dict: config_dict['classes_oracle_id'] = list(config_dict['classes_oracle_id'])
    if 'classes_oracle_ood' in config_dict: config_dict['classes_oracle_ood'] = list(config_dict['classes_oracle_ood'])
    
    with open(os.path.join(save_dir, 'config.json'), 'w') as config_file:
        json.dump(config_dict, config_file, indent=4)

    train_query_dataset = SyntheticHumanPredictionDataset(tr_cntx_sampler.query)
    train_query_loader = DataLoader(train_query_dataset, batch_size=config["train_batch_size"], shuffle=True)

    val_query_dataset = SyntheticHumanPredictionDataset(val_cntx_sampler.query)
    val_query_loader = DataLoader(val_query_dataset, batch_size=config["val_batch_size"], shuffle=False)
    
    # Initialise l2d model (clf and rejector)
    # Load and prepare the model
    model = load_model(config, model_statedict, pretrained_clf_statedict)
    model = model.to(device)
    cudnn.benchmark = True

    if pretrained_clf_statedict is  None:
        print("Pretraining classifier")
        pretrained_base, pretrained_fc = pretrain_clf(model, train_query_loader, val_query_loader, config)
        model.classifier.load_state_dict(pretrained_base)
        model.fc.load_state_dict(pretrained_fc)

    # Set the number of epochs and learning rates
    epochs = num_epochs
    lr = config["lr_wrn"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    iters = 0
    best_epoch = 0
    best_sys_acc = 0
    best_val_loss = float('inf')
    best_val_loss_epoch = 0
    es_patience_counter = 0
    es_patience_counter_loss = 0
    best_val_statedict = None
    best_val_loss_statedict = None

    val_loss_lst = []
    train_loss_lst = []
    
    # Track auxiliary and deferral losses separately
    aux_loss_lst = []
    deferral_loss_lst = []

    clf_acc_by_exp = []
    defer_acc_by_exp = []
    sys_acc_by_exp = []

    # Add auxiliary loss info to banner
    aux_loss_info = ""
    if config.get('aux_loss', False):
        aux_loss_info = f"""
    Auxiliary Loss : Enabled (λ = {config.get('aux_loss_lambda', 1.0)})"""
    else:
        aux_loss_info = """
    Auxiliary Loss : Disabled"""
    
    banner = f"""
    ============================================================
                             TRAINING START
    ============================================================
    Total Epochs   : {epochs}
    Total Batches  : {len(train_query_loader)}
    Batch Size     : {config["train_batch_size"]}
    Learning Rate  : {lr}
    Seed           : {config["seed"]}
    Device         : {device}
    Model          : {config['model']}
    Experiment Name: {config['experiment_name']}{aux_loss_info}
    Save Directory : {save_dir}
    ============================================================
    """
    print(banner)

    def print_epoch_banner(epoch, total_epochs):
        sub_banner = f"""------------------------------------------------------------
                        EPOCH {epoch}/{total_epochs}
------------------------------------------------------------"""
        print(sub_banner)

    for epoch in range(0, epochs):
        print_epoch_banner(epoch, epochs)
        start_time = time.time()
        model.train()
        epoch_train_loss = []
        epoch_deferral_loss = []
        epoch_aux_loss = []

        # For each batch of data
        for batch_n, batch in enumerate(train_query_loader):
            num_batches = len(train_query_loader)
            query_x_batch = batch['xc'][:,0,:,:,:]
            query_exp_pred_batch = batch['mc']
            query_label_batch = batch['yc'][:,0]

            query_x_batch, query_exp_pred_batch, query_label_batch = query_x_batch.to(device), query_exp_pred_batch.to(device), query_label_batch.to(device)
            batch_size = query_x_batch.size()[0]

            # sample context points for training
            expert_cntx = tr_cntx_sampler.sample()
            expert_cntx.xc = expert_cntx.xc.to(device)
            expert_cntx.yc = expert_cntx.yc.to(device)
            expert_cntx.mc = expert_cntx.mc.to(device)


            if (config['model'] == 'ifd'):
                m, weighted_representations, variances = bayesian_inference_per_human(
                    expert_cntx.mc, expert_cntx.yc[0], n_classes, return_variance=True
                )
                outputs = model(query_x_batch, weighted_representations, variances)
                gs = outputs.reshape(n_experts*batch_size, n_classes+1)
                weights_costs = []
                
                for i in range(len(experts_lst)):
                    mu = weighted_representations[i][query_label_batch]
                    if config.lcb:
                        sigma = variances[i][query_label_batch]
                        weights_costs.append((mu-config['lcb_alpha']*sigma).clamp_(min=0))
                    else:
                        weights_costs.append(mu)
                weights_costs = torch.hstack(weights_costs)
                if config.ablation_weights:
                    costs = (m.repeat_interleave(batch_size) == query_label_batch.repeat(n_experts)).int()
                else:
                    costs = weights_costs*(m.repeat_interleave(batch_size) == query_label_batch.repeat(n_experts)).int()
                
                # Main IFD loss (deferral loss)
                deferral_loss = cross_entropy_mod(gs, costs, query_label_batch.repeat(n_experts), n_classes)
                
                # Auxiliary classification loss (only on classifier logits, independent of deferral)
                if config.get('aux_loss', False):
                    # Extract classifier logits from the model (first K classes, before deferral)
                    classifier_logits = model.fc(model.classifier(query_x_batch))  # [B, K]
                    aux_loss = torch.nn.functional.cross_entropy(classifier_logits, query_label_batch)
                    
                    # Combine losses with balancing hyperparameter
                    loss = deferral_loss + config.get('aux_loss_lambda', 1.0) * aux_loss
                    
                    if config.get('debug', False) and batch_n % int(num_batches/4) == 0:
                        print(f"Deferral loss: {deferral_loss.item():.4f}, Auxiliary loss: {aux_loss.item():.4f}, Lambda: {config.get('aux_loss_lambda', 1.0)}")
                else:
                    loss = deferral_loss
        
            elif config['model'] == 'l2d-pop':
                outputs = model(query_x_batch, expert_cntx)
                loss = 0
                for idx_exp, expert in enumerate(experts_lst):
                    m = query_exp_pred_batch[:,idx_exp]
                    costs = (m==query_label_batch).int()
                    loss += cross_entropy_mod(outputs[idx_exp], costs, query_label_batch, n_classes)
                loss /= len(experts_lst)

            elif config['model'] == 'l2d-multi':
                outputs = model(query_x_batch)
                m = (query_exp_pred_batch == query_label_batch.unsqueeze(1)).int()
                loss = cross_entropy_l2dmultiexp(outputs, m, query_label_batch, n_classes)

            elif config['model'] == 'pop-avg':
                outputs = model(query_x_batch) # [B,K+1]
                mode_m = random_mode_per_row(query_exp_pred_batch)
                costs = (mode_m == query_label_batch).int()
                loss = cross_entropy_mod(outputs, costs, query_label_batch, config["n_classes"])
                outputs = outputs.unsqueeze(0).repeat(n_experts,1,1)

            epoch_train_loss.append(loss.item())
            
            # Track loss components for IFD with auxiliary loss
            if config['model'] == 'ifd' and config.get('aux_loss', False):
                epoch_deferral_loss.append(deferral_loss.item())
                epoch_aux_loss.append(aux_loss.item())


            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
            # measure elapsed time
            iters+=1

            if batch_n % int(num_batches/4) == 0:
                print(
                f"Batch {batch_n:2}/{num_batches:<2} | "
                f"Train Loss: {loss.item():7.4f}"
            )

        val_metrics, val_loss = validate(model, val_query_loader, experts_lst, val_cntx_sampler, config)
        train_loss_lst.append(sum(epoch_train_loss)/len(epoch_train_loss))
        
        # Track epoch-level loss components
        if config['model'] == 'ifd' and config.get('aux_loss', False):
            # Store average loss components for this epoch
            deferral_loss_lst.append(sum(epoch_deferral_loss)/len(epoch_deferral_loss))
            aux_loss_lst.append(sum(epoch_aux_loss)/len(epoch_aux_loss))
        else:
            # For other models or without auxiliary loss, just track the main loss
            deferral_loss_lst.append(sum(epoch_train_loss)/len(epoch_train_loss))
            aux_loss_lst.append(0.0)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")

        if config['model'] != 'l2d-multi':
            val_metrics['clf_acc_arr'].pop(-1)
            val_auc_metrics = {f'{key}':compute_auc(val_metrics[key]) for key in val_metrics.keys()}
            clf_acc_by_exp.append(val_auc_metrics['clf_acc_arr'])
            sys_acc_by_exp.append(val_auc_metrics['sys_acc_arr'])
            defer_acc_by_exp.append(val_auc_metrics['exp_acc_arr'])
            val_metrics = val_auc_metrics
        else:
            clf_acc_by_exp.append(val_metrics['clf_acc_arr'])
            sys_acc_by_exp.append(val_metrics['sys_acc_arr'])
            defer_acc_by_exp.append(val_metrics['exp_acc_arr'])


        val_loss_lst.append(val_loss)
        
        # Save metrics including loss components
        metrics_data = {
            'val_clf_acc': clf_acc_by_exp, 
            'val_sys_acc': sys_acc_by_exp, 
            'val_deferral_acc': defer_acc_by_exp,
            'train_loss': train_loss_lst,
            'val_loss': val_loss_lst
        }
        
        # Add loss components for IFD with auxiliary loss
        if config['model'] == 'ifd' and config.get('aux_loss', False):
            metrics_data['deferral_loss'] = deferral_loss_lst
            metrics_data['aux_loss'] = aux_loss_lst
        
        df = pd.DataFrame(data=metrics_data)
        df.to_csv(os.path.join(save_dir, 'metrics_df.csv'))

        # Evaluate early stopping
        if (early_stopping is not None):
            # Check for best validation accuracy
            acc_improved = False
            if val_metrics['sys_acc_arr'] > best_sys_acc:
                best_sys_acc = val_metrics['sys_acc_arr']
                best_val_statedict = deepcopy(model.state_dict())
                torch.save(best_val_statedict,os.path.join(save_dir,'best_val_acc_sd.pth'))
                best_epoch = epoch
                print(f"EARLY STOPPING (ACC): New improvement on val sys acc: {round(best_sys_acc,3)}. Updating best acc params.")
                es_patience_counter = 0
                acc_improved = True
            else:
                print(f"EARLY STOPPING (ACC): No improvement on val sys acc: {round(val_metrics['sys_acc_arr'],3)}. Patience counter: {es_patience_counter+1}/{early_stopping}")
                es_patience_counter += 1
            
            # Check for best validation loss
            loss_improved = False
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_loss_statedict = deepcopy(model.state_dict())
                torch.save(best_val_loss_statedict,os.path.join(save_dir,'best_val_loss_sd.pth'))
                best_val_loss_epoch = epoch
                print(f"EARLY STOPPING (LOSS): New improvement on val loss: {round(best_val_loss,4)}. Updating best loss params.")
                es_patience_counter_loss = 0
                loss_improved = True
            else:
                print(f"EARLY STOPPING (LOSS): No improvement on val loss: {round(val_loss,4)}. Patience counter: {es_patience_counter_loss+1}/{early_stopping}")
                es_patience_counter_loss += 1
            
            # Early stopping triggered by val acc
            if es_patience_counter >= early_stopping and es_patience_counter_loss >= early_stopping:
                print("------"*20)
                # if es_patience_counter == early_stopping:
                #     print(f"EARLY STOPPING: Early stopping at epoch {epoch} due to validation accuracy plateau.")
                #     print(f"Best val acc epoch: {best_epoch}. Best val sys acc: {round(best_sys_acc,3)}")
                # if es_patience_counter_loss == early_stopping:
                #     print(f"EARLY STOPPING: Early stopping at epoch {epoch} due to validation loss plateau.")
                #     print(f"Best val loss epoch: {best_val_loss_epoch}. Best val loss: {round(best_val_loss,4)}")
                print(f"EARLY STOPPING: Early stopping at epoch {epoch}.")
                print(f"Best val loss epoch: {best_val_loss_epoch}. Best val loss: {round(best_val_loss,4)}")
                print(f"Best val acc epoch: {best_epoch}. Best val sys acc: {round(best_sys_acc,3)}")
                print("------"*20)
                break

        if plot_losses==True:
            # Create subplots: 6 for IFD with aux loss, 4 for others
            if config['model'] == 'ifd' and config.get('aux_loss', False):
                fig, ax = plt.subplots(7, figsize=(12,24))
            else:
                fig, ax = plt.subplots(4, figsize=(10,20))
            
            # Main training/validation loss
            ax[0].plot(train_loss_lst, label='Train')
            ax[0].plot(val_loss_lst, label='Val')
            ax[0].vlines(best_epoch, ymin=ax[0].get_ylim()[0], ymax=ax[0].get_ylim()[1], colors='red', linestyles='--', label=f'Best Acc Epoch {best_epoch}')
            ax[0].vlines(best_val_loss_epoch, ymin=ax[0].get_ylim()[0], ymax=ax[0].get_ylim()[1], colors='blue', linestyles='--', label=f'Best Loss Epoch {best_val_loss_epoch}')
            ax[0].legend() 
            ax[0].set_title('Total Loss')

            # Separate loss components for IFD with auxiliary loss
            if config['model'] == 'ifd' and config.get('aux_loss', False):
                # Deferral loss
                ax[1].plot(deferral_loss_lst, label='Deferral Loss', color='red')
                ax[1].set_title('Deferral Loss (IFD)')
                ax[1].legend()
                
                # Auxiliary loss
                ax[2].plot(aux_loss_lst, label='Auxiliary Loss', color='blue')
                ax[2].set_title(f'Auxiliary Loss (λ = {config.get("aux_loss_lambda", 1.0)})')
                ax[2].legend()
                
                # Combined loss components
                ax[3].plot(deferral_loss_lst, label='Deferral', color='red', alpha=0.7)
                ax[3].plot([x * config.get('aux_loss_lambda', 1.0) for x in aux_loss_lst], 
                           label=f'Auxiliary × {config.get("aux_loss_lambda", 1.0)}', color='blue', alpha=0.7)
                ax[3].plot(train_loss_lst, label='Total', color='green', linewidth=2)
                ax[3].set_title('Loss Components')
                ax[3].legend()
                
                # Accuracy plots
                ax[4].plot(clf_acc_by_exp)
                ax[4].plot(pd.Series(clf_acc_by_exp).rolling(50).mean())
                ax[4].set_title('Validation AURCAC')

                ax[5].plot(sys_acc_by_exp)
                ax[5].plot(pd.Series(sys_acc_by_exp).rolling(50).mean())
                ax[5].set_title('Validation AURSAC')

                ax[6].plot(defer_acc_by_exp)
                ax[6].plot(pd.Series(defer_acc_by_exp).rolling(50).mean())
                ax[6].set_title('Validation AURDAC')
                
            else:
                # Standard plots for other models
                ax[1].plot(clf_acc_by_exp)
                ax[1].plot(pd.Series(clf_acc_by_exp).rolling(50).mean())
                ax[1].set_title('Validation AURCAC')

                ax[2].plot(sys_acc_by_exp)
                ax[2].plot(pd.Series(sys_acc_by_exp).rolling(50).mean())
                ax[2].set_title('Validation AURSAC')

                ax[3].plot(defer_acc_by_exp)
                ax[3].plot(pd.Series(defer_acc_by_exp).rolling(50).mean())
                ax[3].set_title('Validation AURDAC')

            plt.tight_layout()   
            fig.savefig(os.path.join(save_dir, 'validation_figs.png'))
            plt.close(fig)  # Close the figure to free memory
    
    # Print final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"Final validation accuracy: {val_metrics['sys_acc_arr']:.4f}")
    print(f"Final validation loss: {val_loss:.4f}")
    print(f"Best validation accuracy: {best_sys_acc:.4f} (epoch {best_epoch})")
    print(f"Best validation loss: {best_val_loss:.4f} (epoch {best_val_loss_epoch})")
    print(f"Saved models:")
    print(f"  - Best accuracy: {os.path.join(save_dir, 'best_val_acc_sd.pth')}")
    print(f"  - Best loss: {os.path.join(save_dir, 'best_val_loss_sd.pth')}")
    print("="*60)

import numpy as np

def extract_expert_predictions_for_sampled_experts(expert_predictions, experts_expertise_classes, 
                                                  cntx_indices, query_indices, y_c, y_tr_q, 
                                                  dataset_name='train', verbose=True):
    """
    Extract expert predictions for specifically sampled experts based on their expertise classes.
    
    Args:
        expert_predictions: List of expert prediction data from ensure_expert_predictions_exist
        experts_expertise_classes: List of expertise classes for sampled experts (from experts_id)
        cntx_indices: Indices for context data
        query_indices: Indices for query data
        y_c: Context labels for verification
        y_tr_q: Query labels for verification
        dataset_name: Which dataset to extract from ('train', 'val', 'test')
        verbose: Whether to print detailed information
        
    Returns:
        m_c: Expert predictions for context data [E, n_c]
        m_q: Expert predictions for query data [E, n_q]
    """
    
    if verbose:
        print("\n" + "="*60)
        print("EXTRACTING EXPERT PREDICTIONS FOR SAMPLED EXPERTS")
        print("="*60)
        print(f"Sampled experts' expertise classes: {experts_expertise_classes}")
    
    # Convert indices to integers if they are tensors
    if hasattr(cntx_indices, 'numpy'):
        cntx_indices_int = cntx_indices.numpy().tolist()
    elif hasattr(cntx_indices, 'tolist'):
        cntx_indices_int = cntx_indices.tolist()
    else:
        cntx_indices_int = [int(idx) for idx in cntx_indices]
    
    if hasattr(query_indices, 'numpy'):
        query_indices_int = query_indices.numpy().tolist()
    elif hasattr(query_indices, 'tolist'):
        query_indices_int = query_indices.tolist()
    else:
        query_indices_int = [int(idx) for idx in query_indices]
    
    if verbose:
        print(f"Context indices type: {type(cntx_indices)}, converted to: {type(cntx_indices_int)}")
        print(f"Query indices type: {type(query_indices)}, converted to: {type(query_indices_int)}")
    
    # Get predictions for the specified dataset
    dataset_predictions = [pred for pred in expert_predictions 
                          if pred['metadata']['dataset_name'] == dataset_name]
    
    # Create mapping from specialty class to expert predictions
    specialty_to_predictions = {}
    for pred_data in dataset_predictions:
        specialty_class = pred_data['metadata']['expert_profile']['specialty_class']
        specialty_to_predictions[specialty_class] = pred_data
    
    if verbose:
        print(f"Available expert specialties: {list(specialty_to_predictions.keys())}")
    
    # Check that we have predictions for all sampled experts
    missing_experts = [exp_class for exp_class in experts_expertise_classes 
                      if exp_class not in specialty_to_predictions]
    if missing_experts:
        print(f"Warning: Missing predictions for expert classes: {missing_experts}")
    
    # Initialize prediction arrays with correct dimensions [E, n_c] and [E, n_q]
    E = len(experts_expertise_classes)  # Number of sampled experts
    n_c = len(cntx_indices_int)  # Number of context samples
    n_q = len(query_indices_int)  # Number of query samples
    
    m_c = np.zeros((E, n_c), dtype=int)  # [E, n_c] - experts x context
    m_q = np.zeros((E, n_q), dtype=int)  # [E, n_q] - experts x query
    
    if verbose:
        print(f"Creating prediction matrices:")
        print(f"m_c shape: [E={E}, n_c={n_c}] = {m_c.shape}")
        print(f"m_q shape: [E={E}, n_q={n_q}] = {m_q.shape}")
    
    # Extract predictions for each sampled expert
    for expert_idx, expertise_class in enumerate(experts_expertise_classes):
        if expertise_class in specialty_to_predictions:
            pred_data = specialty_to_predictions[expertise_class]
            expert_id = pred_data['metadata']['expert_profile']['expert_id']
            predictions = pred_data['predictions']
            
            # Create mapping from sample index to prediction
            prediction_map = {pred['sample_idx']: pred['prediction'] for pred in predictions}
            
            if verbose and expert_idx == 0:  # Debug info for first expert
                print(f"Sample prediction_map keys (first 10): {list(prediction_map.keys())[:10]}")
                print(f"Sample context indices (first 10): {cntx_indices_int[:10]}")
            
            # Extract context predictions for this expert
            for ctx_idx, sample_idx in enumerate(cntx_indices_int):
                if sample_idx in prediction_map:
                    m_c[expert_idx, ctx_idx] = prediction_map[sample_idx]
                else:
                    print(f"Warning: Missing prediction for context sample {sample_idx}")
            
            # Extract query predictions for this expert
            for q_idx, sample_idx in enumerate(query_indices_int):
                if sample_idx in prediction_map:
                    m_q[expert_idx, q_idx] = prediction_map[sample_idx]
                else:
                    print(f"Warning: Missing prediction for query sample {sample_idx}")
            
            if verbose:
                print(f"Expert {expert_idx} (specialty class {expertise_class}, ID: {expert_id}): ✓")
        else:
            print(f"Warning: No predictions found for expert {expert_idx} with specialty class {expertise_class}")
    
    if verbose:
        print(f"\nFinal matrices:")
        print(f"m_c (context expert predictions): {m_c.shape} = [experts, context_samples]")
        print(f"m_q (query expert predictions): {m_q.shape} = [experts, query_samples]")
        
        # Quick verification - calculate accuracy for each expert
        print(f"\nExpert-wise accuracy verification:")
        y_c_np = y_c.numpy() if hasattr(y_c, 'numpy') else y_c
        y_tr_q_np = y_tr_q.numpy() if hasattr(y_tr_q, 'numpy') else y_tr_q
        
        for expert_idx, expertise_class in enumerate(experts_expertise_classes):
            # Context accuracy for this expert
            context_acc = np.mean(m_c[expert_idx] == y_c_np)
            query_acc = np.mean(m_q[expert_idx] == y_tr_q_np)
            
            # Specialty accuracy (how well they do on their specialty class)
            specialty_mask_c = (y_c_np == expertise_class)
            specialty_mask_q = (y_tr_q_np == expertise_class)
            
            if specialty_mask_c.sum() > 0:
                specialty_acc_c = np.mean(m_c[expert_idx][specialty_mask_c] == y_c_np[specialty_mask_c])
            else:
                specialty_acc_c = 0.0
                
            if specialty_mask_q.sum() > 0:
                specialty_acc_q = np.mean(m_q[expert_idx][specialty_mask_q] == y_tr_q_np[specialty_mask_q])
            else:
                specialty_acc_q = 0.0
            
            print(f"Expert {expert_idx} (class {expertise_class}): "
                  f"Overall - C:{context_acc:.3f}, Q:{query_acc:.3f} | "
                  f"Specialty - C:{specialty_acc_c:.3f}, Q:{specialty_acc_q:.3f}")
    
    return torch.tensor(m_c), torch.tensor(m_q)

def extract_expert_predictions_for_validation(expert_predictions, experts_expertise_classes, 
                                            y_val, dataset_name='val', verbose=True):
    """
    Extract expert predictions for validation data (no context/query split).
    
    Args:
        expert_predictions: List of expert prediction data from ensure_expert_predictions_exist
        experts_expertise_classes: List of expertise classes for sampled experts (same as training)
        y_val: Validation labels for verification
        dataset_name: Which dataset to extract from ('val' or 'test')
        verbose: Whether to print detailed information
        
    Returns:
        m_val: Expert predictions for validation data [E, n_val]
    """
    
    if verbose:
        print("\n" + "="*60)
        print(f"EXTRACTING EXPERT PREDICTIONS FOR {dataset_name.upper()} DATA")
        print("="*60)
        print(f"Using same experts with expertise classes: {experts_expertise_classes}")
    
    # Get predictions for the specified dataset
    dataset_predictions = [pred for pred in expert_predictions 
                          if pred['metadata']['dataset_name'] == dataset_name]
    
    # Create mapping from specialty class to expert predictions
    specialty_to_predictions = {}
    for pred_data in dataset_predictions:
        specialty_class = pred_data['metadata']['expert_profile']['specialty_class']
        specialty_to_predictions[specialty_class] = pred_data
    
    if verbose:
        print(f"Available expert specialties: {list(specialty_to_predictions.keys())}")
    
    # Check that we have predictions for all sampled experts
    missing_experts = [exp_class for exp_class in experts_expertise_classes 
                      if exp_class not in specialty_to_predictions]
    if missing_experts:
        print(f"Warning: Missing predictions for expert classes: {missing_experts}")
    
    # Get number of validation samples from first expert's predictions
    if dataset_predictions:
        n_val = len(dataset_predictions[0]['predictions'])
    else:
        raise ValueError(f"No {dataset_name} predictions found!")
    
    # Initialize prediction array [E, n_val]
    E = len(experts_expertise_classes)
    m_val = np.zeros((E, n_val), dtype=int)
    
    if verbose:
        print(f"Creating prediction matrix:")
        print(f"m_{dataset_name} shape: [E={E}, n_{dataset_name}={n_val}] = {m_val.shape}")
    
    # Extract predictions for each sampled expert
    for expert_idx, expertise_class in enumerate(experts_expertise_classes):
        if expertise_class in specialty_to_predictions:
            pred_data = specialty_to_predictions[expertise_class]
            expert_id = pred_data['metadata']['expert_profile']['expert_id']
            predictions = pred_data['predictions']
            
            # Since there's no split, predictions should be in order by sample_idx
            # Sort by sample_idx to ensure correct order
            sorted_predictions = sorted(predictions, key=lambda x: x['sample_idx'])
            
            # Extract all predictions for this expert
            for pred in sorted_predictions:
                sample_idx = pred['sample_idx']
                if sample_idx < n_val:  # Safety check
                    m_val[expert_idx, sample_idx] = pred['prediction']
            
            if verbose:
                print(f"Expert {expert_idx} (specialty class {expertise_class}, ID: {expert_id}): ✓")
        else:
            print(f"Warning: No predictions found for expert {expert_idx} with specialty class {expertise_class}")
    
    if verbose:
        print(f"\nFinal matrix:")
        print(f"m_{dataset_name} (expert predictions): {m_val.shape} = [experts, {dataset_name}_samples]")
        
        # Quick verification - calculate accuracy for each expert
        print(f"\nExpert-wise accuracy verification:")
        y_val_np = y_val.numpy() if hasattr(y_val, 'numpy') else y_val
        
        for expert_idx, expertise_class in enumerate(experts_expertise_classes):
            # Overall accuracy for this expert
            overall_acc = np.mean(m_val[expert_idx] == y_val_np)
            
            # Specialty accuracy (how well they do on their specialty class)
            specialty_mask = (y_val_np == expertise_class)
            
            if specialty_mask.sum() > 0:
                specialty_acc = np.mean(m_val[expert_idx][specialty_mask] == y_val_np[specialty_mask])
                specialty_count = specialty_mask.sum()
            else:
                specialty_acc = 0.0
                specialty_count = 0
            
            print(f"Expert {expert_idx} (class {expertise_class}): "
                  f"Overall: {overall_acc:.3f} | "
                  f"Specialty: {specialty_acc:.3f} ({specialty_count} samples)")
    
    return torch.tensor(m_val)

def main():
    args = parse_args()
    config = set_config(args)
    print("with attn:", config["with_attn"])
    print("with attn:", config["with_attn"])
    print("with attn:", config["with_attn"])
    print("with attn:", config["with_attn"])
    print("with attn:", config["with_attn"])
    set_seed(config["seed"])
    
    if config.get('debug', False):
        print("Debug mode enabled")
        print("Configuration:", config)
    
    train_data, val_data, _, _  = load_dataset(config)
    experts_id, _ = simulate_experts(config, verbose=True)
    X_c, y_c, X_tr_q, y_tr_q, X_val_q, y_val_q, (cntx_indices, query_indices) = get_context_data(config, train_data, val_data)

    _, _, _ = train_and_compute_prototypicality(config)

    from generate_exp_preds import ensure_expert_predictions_exist
    expert_predictions = ensure_expert_predictions_exist(
        config=config,
        specialty_classes=list(range(config.n_classes)),  # All classes
        expert_archetypes=config['expert_archetypes'],
        datasets=['train', 'val', 'test'],
        save_dir='expert_predictions'
    )

    if config.get('debug', False):
        print(f"Experts count: {len(experts_id)}")
        print(f"Context data shape - X_c: {X_c.shape}, y_c: {y_c.shape}")
        print(f"Query data shape - X_tr_q: {X_tr_q.shape}, y_tr_q: {y_tr_q.shape}")

    experts_expertise_classes = [i.classes_oracle[0] for i in experts_id]

    m_c, m_q = extract_expert_predictions_for_sampled_experts(
        expert_predictions=expert_predictions,
        experts_expertise_classes=experts_expertise_classes,
        cntx_indices=cntx_indices,
        query_indices=query_indices,
        y_c=y_c,
        y_tr_q=y_tr_q,
        dataset_name='train',
        verbose=True
    )
    m_val = extract_expert_predictions_for_validation(
        expert_predictions=expert_predictions,
        experts_expertise_classes=experts_expertise_classes,  # Same experts as training
        y_val=y_val_q,  # or whatever your validation labels are called
        dataset_name='val',
        verbose=True
    )

    kwargs = {'num_workers': 0, 'pin_memory': True}
    cntx_sampler_train = ContextSampler(config, context_images=X_c,
                                context_labels=y_c,
                                query_images=X_tr_q,
                                query_labels=y_tr_q,
                                experts_lst=experts_id,
                                n_cntx_pts=config["n_cntx_pts"],
                                context_expert_preds = m_c, # context is always from train
                                query_expert_preds = m_q, # query here is from train
                                seed=None, 
                                device=device, **kwargs)
    cntx_sampler_val = ContextSampler(config, context_images=X_c,
                                        context_labels=y_c,
                                        query_images=X_val_q,
                                        query_labels=y_val_q,
                                        experts_lst=experts_id,
                                        n_cntx_pts=config["n_cntx_pts"], 
                                        context_expert_preds = m_c,
                                        query_expert_preds = m_val, # query here is from val
                                        seed=None,
                                        device=device, **kwargs)
    
    train_model(experts_lst=experts_id, 
                    tr_cntx_sampler=cntx_sampler_train, 
                    val_cntx_sampler=cntx_sampler_val,
                    config=config, num_epochs=config['num_epochs'], plot_losses=True, early_stopping=config['early_stopping'])
    
def main_ood():
    args = parse_args()
    config = set_config(args)
    set_seed(config["seed"])
    
    if config.get('debug', False):
        print("Debug mode enabled")
        print("Configuration:", config)
    
    train_data, val_data, _, _ = load_dataset(config)
    _, experts_ood = simulate_experts(config, verbose=True)
    X_c, y_c, X_tr_q, y_tr_q, X_val_q, y_val_q, (cntx_indices, query_indices) = get_context_data(config, train_data, val_data)

    _, _, _ = train_and_compute_prototypicality(config)

    from generate_exp_preds import ensure_expert_predictions_exist
    expert_predictions = ensure_expert_predictions_exist(
        config=config,
        specialty_classes=list(range(config.n_classes)),  # All classes
        expert_archetypes=config['expert_archetypes'],
        datasets=['train', 'val', 'test'],
        save_dir='expert_predictions'
    )

    if config.get('debug', False):
        print(f"Experts count: {len(experts_ood)}")
        print(f"Context data shape - X_c: {X_c.shape}, y_c: {y_c.shape}")
        print(f"Query data shape - X_tr_q: {X_tr_q.shape}, y_tr_q: {y_tr_q.shape}")

    experts_expertise_classes = [i.classes_oracle[0] for i in experts_ood]

    m_c, m_q = extract_expert_predictions_for_sampled_experts(
        expert_predictions=expert_predictions,
        experts_expertise_classes=experts_expertise_classes,
        cntx_indices=cntx_indices,
        query_indices=query_indices,
        y_c=y_c,
        y_tr_q=y_tr_q,
        dataset_name='train',
        verbose=True
    )
    m_val = extract_expert_predictions_for_validation(
        expert_predictions=expert_predictions,
        experts_expertise_classes=experts_expertise_classes,  # Same experts as training
        y_val=y_val_q,  # or whatever your validation labels are called
        dataset_name='val',
        verbose=True
    )
    
    kwargs = {'num_workers': 0, 'pin_memory': True}
    cntx_sampler_train = ContextSampler(config, context_images=X_c,
                                context_labels=y_c,
                                query_images=X_tr_q,
                                query_labels=y_tr_q,
                                experts_lst=experts_ood,
                                n_cntx_pts=config["n_cntx_pts"],
                                context_expert_preds = m_c, # context is always from train
                                query_expert_preds = m_q, # query here is from train
                                seed=None, 
                                device=device, **kwargs)
    cntx_sampler_val = ContextSampler(config, context_images=X_c,
                                        context_labels=y_c,
                                        query_images=X_val_q,
                                        query_labels=y_val_q,
                                        experts_lst=experts_ood,
                                        n_cntx_pts=config["n_cntx_pts"], 
                                        context_expert_preds = m_c,
                                        query_expert_preds = m_val, # query here is from val
                                        seed=None,
                                        device=device, **kwargs)
    
    train_model(experts_lst=experts_ood, 
                    tr_cntx_sampler=cntx_sampler_train, 
                    val_cntx_sampler=cntx_sampler_val,
                    config=config, num_epochs=config['num_epochs'], plot_losses=True, early_stopping=config['early_stopping'])
    

if __name__ == "__main__":
    args = parse_args()
    if args.mode == "ID":
        main()
    elif args.mode == "OOD":
        main_ood()
    else:
        print(f"Invalid mode: {args.mode}. Please choose 'ID' or 'OOD'.")
