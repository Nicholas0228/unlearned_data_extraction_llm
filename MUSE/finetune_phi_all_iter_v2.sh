master_port=18761
model=phi
lr=1e-5
batch_size=4
gradient_accumulation_steps=4

splits=("full" "full_minus_forget10")


for split in "${splits[@]}"; do
    echo "Processing split: $split"
    CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=$master_port \
    finetune_v2.py --config-name=finetune_v2.yaml split=${split} \
    batch_size=${batch_size} gradient_accumulation_steps=${gradient_accumulation_steps} \
    model_family=${model} lr=${lr} 
done



