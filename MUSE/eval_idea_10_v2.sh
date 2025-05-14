# define variable
port=18765
model_family="phi"
split="forget10"
checkpoint_list=(5553)
precheckpoint_list=(6210)
minus_values=(5.0)


for i in "${!checkpoint_list[@]}"; do
    checkpoint=${checkpoint_list[$i]}
    precheckpoint=${precheckpoint_list[$i]}
    
    for minus in "${minus_values[@]}"; do
        echo "Processing checkpoint: $checkpoint, precheckpoint: $precheckpoint, minus: $minus"
        model_path="../checkpoint_updated/MUSE/final_ft_noLORA_5_epochs_inst_lr1e-05_${model_family}_full_minus_${split}_seed43_1/checkpoint-${checkpoint}"
        pretrained_path="../checkpoint_updated/MUSE/final_ft_noLORA_5_epochs_inst_lr1e-05_${model_family}_full_seed43_1/checkpoint-${precheckpoint}"

        CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port=$port evaluate_util.py \
            model_family=$model_family \
            batch_size=100 \
            split=$split \
            model_path=$model_path \
            +minus_value=$minus \
            +pretrained_path=$pretrained_path \
            --config-name=eval_idea.yaml
    done
done
