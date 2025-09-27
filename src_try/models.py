import copy
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Modified from https://github.com/tung-nd/TNP-pytorch/blob/master/regression/models/attention.py
class MultiHeadAttn(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v, dim_out, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim_out = dim_out
        self.fc_q = nn.Linear(dim_q, dim_out, bias=False)
        self.fc_k = nn.Linear(dim_k, dim_out, bias=False)
        self.fc_v = nn.Linear(dim_v, dim_out, bias=False)
        self.fc_out = nn.Linear(dim_out, dim_out)
        self.ln1 = nn.LayerNorm(dim_out)
        self.ln2 = nn.LayerNorm(dim_out)

    def scatter(self, x):
        return torch.cat(x.chunk(self.num_heads, -1), -3)

    def gather(self, x):
        return torch.cat(x.chunk(self.num_heads, -3), -1)

    def attend(self, q, k, v, mask=None):
        q_, k_, v_ = [self.scatter(x) for x in [q, k, v]]
        A_logits = q_ @ k_.transpose(-2, -1) / math.sqrt(self.dim_out)
        if mask is not None:
            mask = mask.bool().to(q.device)
            mask = torch.stack([mask]*q.shape[-2], -2)
            mask = torch.cat([mask]*self.num_heads, -3)
            A = torch.softmax(A_logits.masked_fill(mask, -float('inf')), -1)
            A = A.masked_fill(torch.isnan(A), 0.0)
        else:
            A = torch.softmax(A_logits, -1)
        return self.gather(A @ v_)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.fc_q(q), self.fc_k(k), self.fc_v(v)
        out = self.ln1(q + self.attend(q, k, v, mask=mask))
        out = self.ln2(out + F.relu(self.fc_out(out)))
        return out

class SelfAttn(MultiHeadAttn):
    def __init__(self, dim_in, dim_out, num_heads=8):
        super().__init__(dim_in, dim_in, dim_in, dim_out, num_heads)

    def forward(self, x, mask=None):
        return super().forward(x, x, x, mask=mask)


def get_activation(act_str):
    if act_str == 'relu':
        return functools.partial(nn.ReLU, inplace=True)
    elif act_str == 'elu':
        return functools.partial(nn.ELU, inplace=True)
    else:
        raise ValueError('invalid activation')

def build_mlp(dim_in, dim_hid, dim_out, depth, activation='relu'):
    act = get_activation(activation)
    if depth==1:
        modules = [nn.Linear(dim_in, dim_out)] # no hidden layers
    else: # depth>1
        modules = [nn.Linear(dim_in, dim_hid), act()]
        for _ in range(depth-2):
            modules.append(nn.Linear(dim_hid, dim_hid))
            modules.append(act())
        modules.append(nn.Linear(dim_hid, dim_out))
    return nn.Sequential(*modules)

class IFD(nn.Module):
    def __init__(self, classifier, num_classes, n_features, dim_hid=128, depth_rej=4):
        super(IFD, self).__init__()
        self.classifier = classifier
        self.fc = nn.Linear(n_features, num_classes)
        self.fc.bias.data.zero_()
        
        # Store configuration flags
        self.num_classes = num_classes
        
        rejector_input_size = 6  # 2 classifier + 2 expert stats + 2 variances
        
        print(f"Initializing IFD with rejector input size: {rejector_input_size}")
        self.decoder = build_mlp(rejector_input_size, dim_hid, 1, depth_rej)
        self.decoder[-1].bias.data.zero_()

        base_mdl_lst = [self.classifier]
        rej_mdl_lst = [self.decoder]
        self.params = nn.ModuleDict({
            'base': nn.ModuleList(base_mdl_lst),
            'clf' : nn.ModuleList([self.fc]),
            'rej': nn.ModuleList(rej_mdl_lst)
        })
        self._initialize_weights()

    def forward(self, x_t, weighted_representations, variances=None):
        n_experts = weighted_representations.shape[0]
        batch_size = x_t.size()[0]

        # get logits of classifier for L2D loss
        x_embed = self.classifier(x_t) # [B,Dx]
        logits_clf = self.fc(x_embed) # [B,K]
        clf_probs = F.softmax(logits_clf, dim=-1)
        clf_preds = torch.argmax(clf_probs, dim=-1)

        logits_clf = logits_clf.unsqueeze(0).repeat(n_experts,1,1) # [E,B,K]

        # predicted expert info
        predicted_experts = weighted_representations.argmax(dim=-1)
        predicted_experts_logits = weighted_representations.max(dim=-1)[0].repeat_interleave(batch_size)
        predicted_experts_clf_logits = torch.stack([clf_probs[batch_n][pred_exp] for pred_exp in predicted_experts for batch_n in range(batch_size)])

        predicted_clf_probs = clf_probs.max(dim=1)[0].repeat(n_experts)
        predicted_clf_probs_representations = torch.stack([weighted_representations[exp_num][clf_probs.max(dim=1)[1]] for exp_num in range(n_experts)]).flatten()

        # Create base rejector inputs
        rejector_inputs_list = [
            predicted_experts_logits,
            predicted_experts_clf_logits,
            predicted_clf_probs,
            predicted_clf_probs_representations
        ]
        
        # Add variance information if available and configured
        if variances is not None:
            # Get variances corresponding to the expert's best class
            variances_experts_best = torch.stack([variances[exp_num][pred_exp] for exp_num, pred_exp in enumerate(predicted_experts)])
            variances_experts_best = variances_experts_best.repeat_interleave(batch_size)
            
            # Get variances corresponding to the classifier's predicted class
            variances_clf_pred = torch.stack([variances[exp_num][clf_preds[batch_n]] for exp_num in range(n_experts) for batch_n in range(batch_size)])
            
            # Add to inputs list
            rejector_inputs_list.append(variances_experts_best)
            rejector_inputs_list.append(variances_clf_pred)
        
        # Stack all inputs
        rejector_inputs = torch.stack(rejector_inputs_list, dim=-1)
        
        # Pass through rejector network
        logits_rej = self.decoder(rejector_inputs) # [E*B,1]

        out = torch.cat([logits_clf, logits_rej.reshape(n_experts, batch_size, 1)], -1)  # [E,B,K+1]        
        out = F.softmax(out, dim=-1)
        return out
    
    def _initialize_weights(self):
        # Initialize weights of linear and embedding layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)  # Xavier initialization for linear layers
                if module.bias is not None:
                    nn.init.zeros_(module.bias)  # Initialize biases to zero
            elif isinstance(module, nn.Embedding):
                nn.init.uniform_(module.weight, -0.1, 0.1)  # Uniform initialization for embedding layer

class Pop_L2D(nn.Module):
    def __init__(self, classifier, num_classes, n_features, dim_hid=128, depth_embed=6, depth_rej=4, dim_class_embed=128,
                 with_attn=False, with_softmax=True, decouple=False):
        super(Pop_L2D, self).__init__()
        self.num_classes = num_classes
        self.with_attn = with_attn
        self.with_softmax = with_softmax
        self.decouple = decouple
        self.dim_hid = dim_hid
        self.n_features = n_features

        self.classifier = classifier
        base_mdl_lst = [self.classifier]
        if self.decouple:
            self.classifier_rej = copy.deepcopy(self.classifier)
            base_mdl_lst += [self.classifier_rej]
        else:
            self.classifier_rej = self.classifier
        
        self.fc = nn.Linear(n_features, num_classes)
        self.fc.bias.data.zero_()

        self.embed_class = nn.Embedding(num_classes, dim_class_embed)    
        self.rejector = build_mlp(n_features+dim_hid, dim_hid, 1, depth_rej)
        self.rejector[-1].bias.data.zero_()

        if not self.with_attn:
            print("L2D-Pop: No attention!!"*20)
            self.embed = build_mlp(n_features+dim_class_embed*2, dim_hid, dim_hid, depth_embed)
        else:
            self.embed = nn.Sequential(
                build_mlp(n_features+dim_class_embed*2, dim_hid, dim_hid, depth_embed-2),
                nn.ReLU(True),
                SelfAttn(dim_hid, dim_hid)
            )
        
        rej_mdl_lst = [self.rejector, self.embed_class, self.embed]

        if with_attn:
            self.attn = MultiHeadAttn(n_features, n_features, dim_hid, dim_hid)
            rej_mdl_lst += [self.attn]
        
        self.params = nn.ModuleDict({
            'base': nn.ModuleList(base_mdl_lst),
            'clf' : nn.ModuleList([self.fc]),
            'rej': nn.ModuleList(rej_mdl_lst)
        })

    def forward(self, x, cntxt=None):
        '''
        Args:
            x : tensor [B,3,32,32]
            cntxt : AttrDict, with entries
                xc : tensor [E,Nc,3,32,32]
                yc : tensor [E,Nc]
                mc : tensor [E,Nc]
        '''
        if cntxt is None:
            n_experts = 1
        else:
            n_experts = cntxt.xc.shape[0]
        
        x_embed = self.classifier(x) # [B,Dx]
        logits_clf = self.fc(x_embed) # [B,K]
        logits_clf = logits_clf.unsqueeze(0).repeat(n_experts,1,1) # [E,B,K]

        if cntxt is None:
            embedding = torch.zeros((n_experts, x.shape[0], self.dim_hid), device=x_embed.device)
        else:
            embedding = self.encode(cntxt, x) # [E,B,H]
        
        x_embed = self.classifier_rej(x) # [B,Dx]
        x_embed = x_embed.unsqueeze(0).repeat(n_experts,1,1) # [E,B,Dx]

        packed = torch.cat([x_embed,embedding], -1) # [E,B,Dx+H]
        
        logit_rej = self.rejector(packed) # [E,B,1]
        
        out = torch.cat([logits_clf,logit_rej], -1) # [E,B,K+1]
        if self.with_softmax:
            out = F.softmax(out, dim=-1)
        return out
    
    def encode(self, cntxt, xt):
        n_experts = cntxt.xc.shape[0]
        batch_size = xt.shape[0]

        cntxt_xc = cntxt.xc.view((-1,) + cntxt.xc.shape[-3:]) # [E*Nc,3,32,32]
        if not self.decouple:
            with torch.no_grad():
                xc_embed = self.classifier_rej(cntxt_xc) # [E*Nc,Dx]
            xc_embed = xc_embed.detach()
        else:
            xc_embed = self.classifier_rej(cntxt_xc) # [E*Nc,Dx]
        
        xc_embed = xc_embed.view(cntxt.xc.shape[:2] + (xc_embed.shape[-1],)) # [E,Nc,Dx]

        yc_embed = self.embed_class(cntxt.yc) # [E,Nc,H]
        mc_embed = self.embed_class(cntxt.mc) # [E,Nc,H]
        out = torch.cat([xc_embed,yc_embed,mc_embed], -1) # [E,Nc,Dx+2H]

        out = self.embed(out) # [E,Nc,H]

        if not self.with_attn:
            embedding = out.mean(-2) # [E,H]
            embedding = embedding.unsqueeze(1).repeat(1,batch_size,1) # [E,B,H]
        else:
            xt_embed = self.classifier_rej(xt) # [B,Dx]
            if not self.decouple:
                xt_embed = xt_embed.detach() # stop gradients flowing
            xt_embed = xt_embed.unsqueeze(0).repeat(n_experts,1,1) # [E,B,Dx]
            embedding = self.attn(xt_embed, xc_embed, out) # [E,B,H]
        
        return embedding
    
class ClassifierRejector(nn.Module):
    def __init__(self, classifier, num_classes, n_features, with_softmax=True, decouple=False):
        super(ClassifierRejector, self).__init__()
        self.classifier = classifier
        base_mdl_lst = [self.classifier]
        if decouple:
            self.classifier_rej = copy.deepcopy(self.classifier)
            base_mdl_lst += [self.classifier_rej]
        else:
            self.classifier_rej = self.classifier

        self.fc = nn.Linear(n_features, num_classes)
        self.fc.bias.data.zero_()

        self.fc_rej = nn.Linear(n_features, 1)
        self.fc_rej.bias.data.zero_()

        self.with_softmax = with_softmax
        self.params = nn.ModuleDict({
            'base': nn.ModuleList(base_mdl_lst),
            'clf' : nn.ModuleList([self.fc,self.fc_rej])
        })

    def forward(self, x):
        out = self.classifier(x)
        logits_clf = self.fc(out) # [B,K]

        out = self.classifier_rej(x)
        logit_rej = self.fc_rej(out) # [B,1]

        out = torch.cat([logits_clf,logit_rej], -1) # [B,K+1]

        if self.with_softmax:
            out = F.softmax(out, dim=-1)
        return out

class ClassifierRejectorMulti(nn.Module):
    def __init__(self, classifier, num_classes, num_experts, n_features, with_softmax=True, decouple=False):
        super(ClassifierRejectorMulti, self).__init__()
        self.classifier = classifier
        self.num_experts = num_experts
        base_mdl_lst = [self.classifier]
        if decouple:
            self.base_model_rej = copy.deepcopy(self.classifier)
            base_mdl_lst += [self.base_model_rej]
        else:
            self.base_model_rej = self.classifier

        self.fc = nn.Linear(n_features, num_classes)
        self.fc.bias.data.zero_()

        self.fc_rej = nn.Linear(n_features, num_experts)
        self.fc_rej.bias.data.zero_()

        self.with_softmax = with_softmax
        self.params = nn.ModuleDict({
            'base': nn.ModuleList(base_mdl_lst),
            'clf' : nn.ModuleList([self.fc,self.fc_rej])
        })

    def forward(self, x):
        out = self.classifier(x)
        logits_clf = self.fc(out) # [B,K]

        out = self.base_model_rej(x)
        logit_rej = self.fc_rej(out) # [B,E]

        out = torch.cat([logits_clf,logit_rej], -1) # [B,K+E]

        if self.with_softmax:
            out = F.softmax(out, dim=-1)
        return out