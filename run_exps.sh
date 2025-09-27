#!/bin/bash

for dataset in 'organs_axial'
do
    for seed in 1 2 3 4 5
    do
        # Run for "realistic_specialist"
        python src_try/main.py --model 'l2d-multi' --dataset $dataset --experiment_name 'new_expert' --seed $seed --lcb --expert_archetypes "realistic_specialist" --early_stopping 50 
        python src_try/main.py --model 'l2d-pop' --dataset $dataset --experiment_name 'new_expert' --seed $seed --lcb --expert_archetypes "realistic_specialist" --early_stopping 50
        python src_try/main.py --model 'ifd' --dataset $dataset --experiment_name 'new_expert_0' --seed $seed --lcb --expert_archetypes "realistic_specialist" --aux_loss --aux_loss_lambda 0 --early_stopping 50
        python src_try/main.py --model 'ifd' --dataset $dataset --experiment_name 'new_expert_5' --seed $seed --lcb --expert_archetypes "realistic_specialist" --aux_loss --aux_loss_lambda 5 --early_stopping 50

        # Run for "variable_specialist"
        python src_try/main.py --model 'l2d-multi' --dataset $dataset --experiment_name 'new_expert' --seed $seed --lcb --expert_archetypes "variable_specialist" --early_stopping 50 
        python src_try/main.py --model 'l2d-pop' --dataset $dataset --experiment_name 'new_expert_var' --seed $seed --lcb --expert_archetypes "variable_specialist" --early_stopping 50
        python src_try/main.py --model 'ifd' --dataset $dataset --experiment_name 'new_expert_var_0' --seed $seed --lcb --expert_archetypes "variable_specialist" --aux_loss --aux_loss_lambda 0 --early_stopping 50
        python src_try/main.py --model 'ifd' --dataset $dataset --experiment_name 'new_expert_var_5' --seed $seed --lcb --expert_archetypes "variable_specialist" --aux_loss --aux_loss_lambda 5 --early_stopping 50
    done
done
