#step1: generate training json
JSON_DIR='./data/train_json'
python ./data/create_dataset_aff.py --folder_dir ../data_collection/data/train_data --output_dir "$JSON_DIR" --num_point 20

#step2: train model
OUTPUT_DIR='./exp/train_ckpts_0613_cat'
mkdir -p "$OUTPUT_DIR"
CUDA_VISIBLE_DEVICES=0 python -u -m torch.distributed.launch --master_port=11712 --nproc_per_node=1 --use_env main_finetune.py --batch_size 1 \
      --epochs 10 --warmup_epochs 1 --blr 1e-3 --weight_decay 0.02 \
      --output_dir "$OUTPUT_DIR" \
      --pretrained_path ./ckpts/BIAS_LORA_NORM-336-Chinese-7B.pth \
      --llama_path ./ckpts/llama_model_weights \
      --bins True \
      --mlm True\
      --aff_prior \
      --data_config "$JSON_DIR"