# generate around 20,000 training samples, then stop it manually
python gen_offline_data.py \
  --data_dir ../data/train_data\
  --data_fn ../stats/train_id.txt\
  --primact_types pulling \
  --num_processes 10 \
  --num_epochs 200 \
  --starting_epoch 0 \
  --ins_cnt_fn ../stats/ins_cnt_46cats.txt \
  --mode train 

# delete the extra testing dataset, and remain around 1,500 testing samples. Make sure that each category has as least 50 samples.
python gen_offline_data.py \
  --data_dir ../data/test_data\
  --data_fn ../stats/test_id.txt\
  --primact_types pulling \
  --num_processes 10 \
  --num_epochs 20 \
  --starting_epoch 0 \
  --ins_cnt_fn ../stats/ins_cnt_46cats.txt \
  --mode test 
