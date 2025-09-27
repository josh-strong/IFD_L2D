import random
import numpy as np

class SyntheticExpertOverlap():
    def __init__(self, classes_oracle, n_classes=10, p_in=1.0, p_out=0.1):
        self.expert_static = True
        self.classes_oracle = classes_oracle
        if isinstance(self.classes_oracle, int):
            self.classes_oracle = [self.classes_oracle]
        self.n_classes = n_classes
        self.p_in = p_in
        self.p_out = p_out

    def __call__(self, labels, seed=None):
        if seed is not None:
            np_random_state = np.random.RandomState(seed)
            random_state = random.Random(seed)
        else:
            np_random_state = np.random
            random_state = random

        batch_size = labels.size()[0]
        outs = [0] * batch_size
        for i in range(batch_size):
            if labels[i].item() in self.classes_oracle:
                coin_flip = np_random_state.binomial(1, self.p_in)
                if coin_flip == 1:
                    outs[i] = labels[i].item()
                else:
                    outs[i] = random_state.randint(0, self.n_classes - 1)
            else:
                coin_flip = np_random_state.binomial(1, self.p_out)
                if coin_flip == 1:
                    outs[i] = labels[i].item()
                else:
                    outs[i] = random_state.randint(0, self.n_classes - 1)
        
        return outs

def simulate_experts(config, verbose=False):
    seed = config["seed"]
    random.seed(seed)

    experts_id = []
    experts_ood = []                                                                                                                        

    sampled_ID_experts = random.sample(range(config['n_classes']), config['n_ID_experts'])
    OOD_experts = [num for num in range(config["n_classes"]) if num not in sampled_ID_experts]

    for i in sampled_ID_experts:
        class_oracle = i
        print(class_oracle)
        expert = SyntheticExpertOverlap(classes_oracle=class_oracle, n_classes=config["n_classes"], p_in=config['p_in'], p_out=config['p_out'])
        experts_id.append(expert)

    for i in OOD_experts:
        class_oracle = i
        expert = SyntheticExpertOverlap(classes_oracle=class_oracle, n_classes=config["n_classes"], p_in=config['p_in'], p_out=config['p_out'])
        experts_ood.append(expert)

    if verbose:
        print(f"In-distribution experts: {[i.classes_oracle[0] for i in experts_id]}")
        print(f"Out-of-distribution experts: {[i.classes_oracle[0] for i in experts_ood]}")

    return experts_id, experts_ood