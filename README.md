# ManipLLM
The official codebase for ManipLLM:  Embodied Multimodal Large Language Model for Object-Centric Robotic Manipulation (CVPR 2024)

## Acknowledgement
This repo benefits from [LLama_Adapter](https://github.com/OpenGVLab/LLaMA-Adapter) and [Where2act](https://github.com/daerduoCarey/where2act). Thanks for their wonderful works.

## Setup
conda create --name manipllm python=3.8
conda activate manipllm
pip install -r requirements.txt

            
## Data Collection
- Download OUR train/test data:
  
  [Train data downloading](https://disk.pku.edu.cn/link/AAC241C979556F4FBAB8B1895B2E63E9AA) ....
  
  [Test data downloading](https://disk.pku.edu.cn/link/AA103C5B00398E4E4089903CB06AC09D8C) ....
  
  
  The downloaded 'train_data' and 'test_data' folder should be unziped under /ManipLLM/data_collection/data.

- Collect data by your own: Download [partnet mobility](https://sapien.ucsd.edu/downloads) urdf from its official website and place under ./ManipLLM/data_collection/asset.
  ```bash
  ./asset/original_sapien_dataset
    ├── 148
    |   └── mobility.urdf
    ├── 149
    |   └── mobility.urdf
    ├── ...
    │   ...
    └── ...
  
  cd ./ManipLLM/data_collection/code
  
  bash scripts/run_gen_offline_data.sh

This command will first generate training dataset and then generate the testing dataset.

## Model Training
- Preparation:

  Download checkpoints for [CLIP](https://disk.pku.edu.cn/link/AA93FF7210CF0D4F428850C0F520C81453), [LLaMa-Adapter](https://disk.pku.edu.cn/link/AA682A19DB7FDA4028B112449D24BBC308). The downloaded checkpoints should be placed under /ManipLLM/train/ckpts. Obtain the LLaMA backbone weights using this [form](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform). Please note that checkpoints from unofficial sources (e.g., BitTorrent) may contain malicious code and should be used with care. Organize the downloaded checkpoints in the following structure:
    ```plaintext
    ./ckpts/llama_model_weights
    ├── 7B
    │   ├── checklist.chk
    │   ├── consolidated.00.pth
    │   └── params.json
    └── tokenizer.model
    ./ckpts/BIAS_LORA_NORM-336-Chinese-7B.pth
    ./ckpts/ViT-L-14-336px.pt
- Model training: The training requires the server to has a least 40g memory. The command will first generate the training json, then start training

  
  ```bash
  cd ./ManipLLM/train
  
  bash finetune.sh

## Model Testing
The public code only infers on the final prompt without chain-of-thought, predicting the pose directly. 

Remember to add the checkpoints of [CLIP](https://disk.pku.edu.cn/link/AA93FF7210CF0D4F428850C0F520C81453), [LLaMa](same with the process in training), and [LLaMa-Adapter](https://disk.pku.edu.cn/link/AA682A19DB7FDA4028B112449D24BBC308) under /ManipLLM/test/ckpts as well.

Place the released [ManipLLM](https://disk.pku.edu.cn/link/AAD623030E00724891BA258A8575D434C8) checkpoint under /ManipLLM/test/exp or use your own trained checkpoint.

The testing requires the server to has a least 40g memory. This command will first use the model to infer on all the test samples, and then interact with object in the simulator (SAPIEN).
  
  ```bash
  cd ./ManipLLM/test
  
  bash test.sh

