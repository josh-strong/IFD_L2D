import torch
from torchvision.models import efficientnet_b0
from wideresnet import WideResNetBase
from resnet224 import ResNet34
from resnet import resnet20
from models import IFD, ClassifierRejector, ClassifierRejectorMulti, Pop_L2D
from datasets import load_ham10000, load_bus, load_organs_axial, load_bloodmnist, load_oct, load_new_ham10000, load_cifar10_new, load_imagenet16_greyscale
import pandas as pd
import numpy as np
from utils import ROOT
import os
import torch.nn.functional as F
from torchvision import models
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(config, state_dict=None, pretrained_clf_statedict=None):
    if config['dataset'] in ['ham10000', 'new_ham10000']:
        clf_base = ResNet34()
        n_features = clf_base.n_features
    elif config['dataset'] == 'cifar10h':
        clf_base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Use backbone features (pre-logits) as classifier output
        n_features = 1000
    elif config['dataset'] == 'imagenet16_grey':
        clf_base = ResNet34()
        conv1 = clf_base.resnet.conv1
        clf_base.resnet.conv1 = torch.nn.Conv2d(1, conv1.out_channels, kernel_size=conv1.kernel_size, stride=conv1.stride, padding=conv1.padding, bias=conv1.bias)
        n_features = clf_base.n_features
    elif config['dataset'] in ['bus','organs_axial']:
        clf_base = efficientnet_b0(weights=True)
        clf_base.features[0][0] = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        n_features = clf_base.classifier[1].out_features
    elif config['dataset'] == 'blood_mnist':
        clf_base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        clf_base.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        n_features = clf_base.classifier[-1].out_features  # Extract features before FC
    elif config['dataset'] == 'oct':
        clf_base = efficientnet_b0(pretrained=True)
        clf_base.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        n_features = clf_base.classifier[1].out_features
    else:
        raise ValueError(f"Unknown dataset: {config['dataset']}")
    if pretrained_clf_statedict is not None:
        clf_base.load_state_dict(pretrained_clf_statedict)

    if config['model'] == 'ifd':
        model = IFD(
            classifier=clf_base, 
            num_classes=int(config["n_classes"]), 
            n_features=n_features,
            dim_hid=config["dim_hid"],
            depth_rej=config["depth_reject"]
        )

    elif config['model'] == 'l2d-pop':
        model =  Pop_L2D(clf_base, num_classes=int(config["n_classes"]), n_features=n_features, \
                                                            with_attn=config["with_attn"], with_softmax=config["with_softmax"], decouple=config["decouple"], \
                                                            depth_embed=config["depth_embed"], depth_rej=config["depth_reject"]) 

    elif config['model'] == 'l2d-multi':
        model = ClassifierRejectorMulti(clf_base, num_classes=int(config["n_classes"]), num_experts=int(config["n_experts"]), n_features=n_features)

    elif config['model'] == 'pop-avg':
        model = ClassifierRejector(clf_base, num_classes=int(config["n_classes"]), n_features=n_features)

    if state_dict is not None:
        model.load_state_dict(state_dict)

    model.to('cuda')
    return model

def load_dataset(config):
    if config["dataset"] == 'ham10000':
        config["n_classes"] = 7
        config["n_ID_experts"] = 3
        train_data, val_data, test_data, expert_data = load_ham10000()
    elif config["dataset"] == 'new_ham10000':
        config["n_classes"] = 7
        config["n_ID_experts"] = 3
        config["n_cntx_pts"] = 35
        train_data, val_data, test_data, expert_data = load_new_ham10000()
    elif config["dataset"] == 'bus':
        config['n_ID_experts'] = 1
        config["n_classes"] = 3
        config["n_cntx_pts"] = 35
        train_data, val_data, test_data = load_bus(os.path.join(ROOT, 'data/bus/bus_data/bus'))
    elif config["dataset"] == 'organs_axial':
        config['n_ID_experts'] = 5
        config["n_classes"] = 11
        config["n_cntx_pts"] = 35
        train_data, val_data, test_data, expert_data = load_organs_axial(os.path.join(ROOT, 'data/organs_axial/organs_axial_data/organs_axial'))
    elif config["dataset"] == 'blood_mnist':
        config['n_ID_experts'] = 4
        config["n_classes"] = 10
        config["n_cntx_pts"] = 35
        train_data, val_data, test_data, expert_data = load_bloodmnist()
    elif config["dataset"] == 'oct':
        config['n_ID_experts'] = 2
        config["n_classes"] = 4
        config["n_cntx_pts"] = 10
        train_data, val_data, test_data, expert_data = load_oct()
    elif config["dataset"] == 'cifar10h':
        config['n_ID_experts'] = 5
        config["n_classes"] = 10
        config["n_cntx_pts"] = 25  
        train_data, val_data, test_data, expert_data = load_cifar10_new(seed=config.get('seed', 42), max_total_images=10000)
    elif config["dataset"] == 'imagenet16_grey':
        config['n_ID_experts'] = 5 
        config["n_classes"] = 16
        config["n_cntx_pts"] = 35
        train_data, val_data, test_data, expert_data = load_imagenet16_greyscale()
    else:
        raise ValueError(f"Unknown dataset: {config['dataset']}")
    return train_data, val_data, test_data, expert_data

def bayesian_inference_per_human(predictions, ground_truth, K, device=device, alpha_prior=None, beta_prior=None, return_posterior_params=False, return_variance=False):
        # If no prior means are provided, assume an uninformative Beta(1,1)
        if alpha_prior is None:
            alpha_prior = torch.ones(K)  # Uninformative Beta(1,1) prior
            beta_prior = torch.ones(K)
        else:
            # Convert prior means into Beta parameters
            alpha_prior = alpha_prior
            beta_prior = beta_prior

        n_humans = predictions.size(0)
        expert_classes = []
        human_posterior_means = []
        alpha_post_lst = []
        beta_post_lst = []
        human_posterior_variances = [] # Store variances

        for human in range(n_humans):
            human_predictions = predictions[human]

            # Count correct and total predictions per class
            correct_counts = torch.zeros(K)
            total_counts = torch.zeros(K)
            for label, prediction in zip(ground_truth, human_predictions):
                total_counts[label] += 1
                if label == prediction:
                    correct_counts[label] += 1

            # Update posterior with prior
            alpha_post = alpha_prior + correct_counts
            beta_post = beta_prior + total_counts - correct_counts
            posterior_means = alpha_post / (alpha_post + beta_post)
            human_posterior_means.append(posterior_means)
            alpha_post_lst.append(alpha_post)
            beta_post_lst.append(beta_post)
            
            # Calculate posterior variances using beta distribution variance formula
            posterior_variances = (alpha_post * beta_post) / ((alpha_post + beta_post)**2 * (alpha_post + beta_post + 1))
            # Handle potential division by zero (if alpha_post + beta_post is too small)
            posterior_variances = torch.where(
                torch.isnan(posterior_variances) | torch.isinf(posterior_variances),
                torch.ones_like(posterior_variances),  # Set to 1 (maximum variance for Beta) if NaN or Inf
                posterior_variances
            )
            human_posterior_variances.append(posterior_variances)

            # Select expert class
            expert_class = torch.argmax(posterior_means).item()
            expert_classes.append(expert_class)

        if return_posterior_params:
            return torch.tensor(expert_classes).to(device), torch.stack(human_posterior_means).to(device), alpha_post_lst, beta_post_lst
        elif return_variance:
            return torch.tensor(expert_classes).to(device), torch.stack(human_posterior_means).to(device), torch.stack(human_posterior_variances).to(device)
        else:
            return torch.tensor(expert_classes).to(device), torch.stack(human_posterior_means).to(device)

def random_mode_per_row(tensor, seed=None):
    N, D = tensor.shape
    counts = torch.zeros(N, tensor.max() + 1, device=tensor.device, dtype=torch.long)
    counts.scatter_add_(1, tensor, torch.ones_like(tensor, dtype=torch.long))
    mask = counts == counts.max(dim=1, keepdim=True).values
    if seed is not None:
        generator = torch.Generator(device=tensor.device).manual_seed(seed)
    else:
        generator = None
    rand_vals = torch.rand(counts.shape, device=counts.device, generator=generator)
    rand_vals[~mask] = float('-inf')
    return rand_vals.argmax(dim=1)

def cross_entropy_mod(outputs, m, labels, n_classes, eps=1e-10):
    '''
    The L_{CE} loss implementation for CIFAR with alpha=1
    ----
    outputs: network outputs
    m: cost of deferring to expert cost of classifier predicting (I_{m =y})
    labels: target
    n_classes: number of classes
    '''
    batch_size = outputs.size()[0]
    rc = [n_classes] * batch_size # idx to extract rejector function
    outputs = -m * torch.log2(outputs[range(batch_size), rc]+eps) - torch.log2(outputs[range(batch_size), labels])
    return torch.sum(outputs) / batch_size

def cross_entropy_l2dmultiexp(outputs, m, labels, n_classes):
    '''
    The L_{CE} loss implementation for CIFAR with alpha=1
    ----
    outputs: network outputs
    m: cost of deferring to expert cost of classifier predicting (I_{m =y})
    labels: target
    n_classes: number of classes
    '''
    batch_size = outputs.size()[0]
    clf_loss = -torch.log2(outputs[range(batch_size),labels])
    rej_loss = (-m * torch.log2(outputs[range(batch_size),n_classes:])).sum(axis=-1)
    tot_loss = clf_loss + rej_loss
    return torch.sum(tot_loss)/batch_size

def compute_auc(accuracy_array):
    """
    Computes the area under the curve (AUC) for system accuracy using the trapezoidal rule.

    Parameters:
        accuracy_array (array-like): An array of accuracy values.

    Returns:
        float: The computed area under the curve.
    """
    # Ensure the input is a numpy array
    accuracy_array = np.array(accuracy_array)

    # Compute the AUC using the trapezoidal rule
    auc = np.trapz(accuracy_array)
    
    return auc/len(accuracy_array)

def validate(model, data_query_loader, experts_lst, cntx_sampler, config):
    n_classes = config['n_classes']
    n_experts = len(experts_lst)
    model.eval()

    with torch.no_grad():
        labels_lst = []
        all_outputs = []
        all_preds_lst = []
        for i, batch in enumerate(data_query_loader):
            query_x_batch = batch['xc'][:,0,:,:,:]
            query_exp_pred_batch = batch['mc']
            query_label_batch = batch['yc'][:,0]

            batch_size = query_x_batch.size()[0]
            n_experts = len(experts_lst)
            labels_lst.extend(query_label_batch.cpu().numpy())
            query_x_batch, query_exp_pred_batch, query_label_batch = query_x_batch.to(device), query_exp_pred_batch.to(device), query_label_batch.to(device)

            # Sample context points
            expert_cntx = cntx_sampler.sample()
            expert_cntx.xc = expert_cntx.xc.to(device)
            expert_cntx.yc = expert_cntx.yc.to(device)
            expert_cntx.mc = expert_cntx.mc.to(device)

            # get L2D predictions. Logits of clf + human prediction
            if config['model'] == 'ifd':
                m, weighted_representations, variances = bayesian_inference_per_human(
                    expert_cntx.mc, expert_cntx.yc[0], n_classes, return_variance=True
                )
                outputs = model(query_x_batch, weighted_representations, variances)
                gs = outputs.reshape(n_experts*batch_size, n_classes+1)
                weights_costs = []
                for i in range(len(experts_lst)):
                    mu = weighted_representations[i][query_label_batch]
                    if config['lcb']:
                        sigma = variances[i][query_label_batch]
                        weights_costs.append((mu-config['lcb_alpha']*sigma).clamp_(min=0))
                    else:
                        weights_costs.append(mu)
                weights_costs = torch.hstack(weights_costs)
                costs = weights_costs*(m.repeat_interleave(batch_size) == query_label_batch.repeat(n_experts)).int()
                deferral_loss = cross_entropy_mod(gs, costs, query_label_batch.repeat(n_experts), n_classes)
                
                # Add auxiliary loss if enabled
                if config.get('aux_loss', False):
                    classifier_logits = model.fc(model.classifier(query_x_batch))
                    aux_loss = torch.nn.functional.cross_entropy(classifier_logits, query_label_batch)
                    loss = deferral_loss + config.get('aux_loss_lambda', 1.0) * aux_loss
                else:
                    loss = deferral_loss

            elif config['model'] == 'pop-avg':
                outputs = model(query_x_batch) # [B,K+1]
                mode_m = random_mode_per_row(query_exp_pred_batch)
                costs = (mode_m == query_label_batch).int()
                loss = cross_entropy_mod(outputs, costs, query_label_batch, config["n_classes"])
                outputs = outputs.unsqueeze(0).repeat(n_experts,1,1)

            elif config['model'] == 'l2d-pop':
                outputs = model(query_x_batch, expert_cntx) # outputs g_1, ..., g_k, g_{\perp} for L2D loss
                loss = 0
                # for each expert in training
                for idx_exp, expert in enumerate(experts_lst):
                    m = query_exp_pred_batch[:,idx_exp]
                    costs = (m==query_label_batch).int()
                    loss += cross_entropy_mod(outputs[idx_exp], costs, query_label_batch, n_classes)
                loss /= len(experts_lst)

            elif config['model'] == 'l2d-multi':
                outputs = model(query_x_batch)
                m = (query_exp_pred_batch == query_label_batch.unsqueeze(1)).int()
                loss = cross_entropy_l2dmultiexp(outputs, m, query_label_batch, n_classes)
                all_preds_lst.append(query_exp_pred_batch)

            all_outputs.append(outputs)


    if config["model"] != 'l2d-multi':
        # multiple experts to choose from
        m_outputs = torch.concat(all_outputs, dim=1)
        df_lst = []
        for dim in range(m_outputs.size()[0]):
            clf_preds = m_outputs[dim][:,:n_classes].argmax(dim=-1).cpu()
            def_probs = m_outputs[dim][:,-1].cpu()
            def_conf = (m_outputs[dim][:,:n_classes].max(dim=-1)[0] - m_outputs[dim][:,-1]).cpu()
            def_flag = (m_outputs[dim][:,:n_classes].max(dim=-1)[0] < m_outputs[dim][:,-1]).int().cpu()
            df = pd.DataFrame(data={'clf_preds':clf_preds,  
                                    'def_probs':def_probs, 
                                    'def_conf':def_conf,
                                    'def_flag':def_flag,
                                    'labels':data_query_loader.dataset.yc[0],
                                    'exp_preds':data_query_loader.dataset.mc[dim]})
            df_lst.append(df)

        df_a = pd.concat([df.reset_index() for df in df_lst])
        df_a.reset_index(drop=True,inplace=True)
        df_a.sort_values(by='def_conf')
        df_a = df_a.sort_values(by=['index', 'def_conf', 'def_probs'])
        df_a = df_a.drop_duplicates(subset=['index'], keep='first')

        clf_acc_arr = []
        exp_acc_arr = []
        sys_acc_arr = []
        df_a = df_a.sort_values(by='def_conf')
        for i in range(len(df_a)):
            clf_df = df_a.iloc[i+1:]
            e_df = df_a.iloc[:i+1]
            clf_acc = (clf_df.clf_preds == clf_df.labels).mean()
            exp_acc = (e_df.exp_preds == e_df.labels).mean()
            sys_acc = ((e_df.exp_preds == e_df.labels).sum() + (clf_df.clf_preds == clf_df.labels).sum())/len(df_a)
            clf_acc_arr.append(clf_acc)
            exp_acc_arr.append(exp_acc)
            sys_acc_arr.append(sys_acc)
        dict = {'clf_acc_arr':clf_acc_arr,
                'exp_acc_arr':exp_acc_arr,
                'sys_acc_arr':sys_acc_arr,
                }
        return dict, loss.item()
    else:
        outputs = torch.vstack(all_outputs)
        all_preds_lst = torch.vstack(all_preds_lst)
        clf_probs, clf_preds = outputs[:,:config["n_classes"]].max(dim=-1)
        defer_probs, defer_preds = outputs[:,config["n_classes"]:].max(dim=-1)
        defer_conf = defer_probs - clf_probs
        defer_flag = clf_probs < defer_probs

        clf_acc = (clf_preds[defer_flag == False] == torch.tensor(labels_lst).cuda()[defer_flag == False]).float().mean()
        defer_acc = (all_preds_lst[range(len(all_preds_lst)), defer_preds][defer_flag==True] == torch.tensor(labels_lst).cuda()[defer_flag == True]).float().mean()
        sys_acc = ((clf_preds[defer_flag == False] == torch.tensor(labels_lst).cuda()[defer_flag == False]).sum() + (all_preds_lst[range(len(all_preds_lst)), defer_preds][defer_flag==True] == torch.tensor(labels_lst).cuda()[defer_flag == True]).sum())/len(clf_preds)

        metrics = {
        'clf_acc_arr':clf_acc.item(),
        'exp_acc_arr':defer_acc.item(),
        'sys_acc_arr':sys_acc.item()
        }
    return metrics, loss.item()
