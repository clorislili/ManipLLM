#step1: model inference 
OUPUT_DIR='./test_result/train_0612_e9_0613'
# CUDA_VISIBLE_DEVICES=2 python test_llama.py \
#   --llama_dir ./ckpts/llama_model_weights \
#   --adapter_dir ../train/exp/train_ckpts_2/checkpoint-9.pth \
#   --data_dir ../data_collection/data/test_data \
#   --out_dir "$OUPUT_DIR" \
#   --action pulling

#step2: test in simulator
# python test_entireprocess_in_sapien.py \
#   --data_dir /vepfs-cnsh4137610c2f4c/algo/user8/lixiaoqi/cloris/ManipLLM0604/data_collection/data/test_data \
#   --num_processes 10 \
#   --out_dir "$OUPUT_DIR" \
#   --no_gui

# #step3: calculate success rate
python cal_test_mani_succ_rate.py \
    --primact_type pulling \
    --data_dir "$OUPUT_DIR"