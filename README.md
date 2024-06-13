# ManipLLM
The official codebase for ManipLLM:  Embodied Multimodal Large Language Model for Object-Centric Robotic Manipulation (CVPR 2024)

## Data Collection
- Download OUR train/test data:
  
  [Train data downloading](URL) ....
  
  [Test data downloading](URL) ....
  
  
  The downloaded 'train_data' and 'test_data' folder should be placed under /ManipLLM/data_collection/data.

- Collect data by your own: Download [partnet mobility](https://sapien.ucsd.edu/downloads) urdf from its official website and place under ./ManipLLM/data_collection/asset.
  ```bash
  cd ./ManipLLM/data_collection/code
  
  bash scripts/run_gen_offline_data.sh

## Model Training
- Preparation:

  Download checkpoints for [CLIP](https://drive.google.com/file/d/1XxfRxUL442Zh4NN0-JeIyNSJnLlTpxsu/view?usp=sharing), [LLaMa-Adapter](https://drive.google.com/file/d/1JxxoLhV9lbS4iQNALU8vwUrRReGxS0eU/view?usp=sharing). The downloaded checkpoint should be placed under /ManipLLM/train/ckpts. Obtain the LLaMA backbone weights using this [form](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform). Please note that checkpoints from unofficial sources (e.g., BitTorrent) may contain malicious code and should be used with care. Organize the downloaded file in the following structure:
    ```plaintext
    ./ckpts/llama_model_weights
    ├── 7B
    │   ├── checklist.chk
    │   ├── consolidated.00.pth
    │   └── params.json
    └── tokenizer.model
    ./ckpts/BIAS_LORA_NORM-336-Chinese-7B.pth
    ./ckpts/ViT-L-14-336px.pt
- Model training: it first generates the training json, then start training
  ```bash
  cd ./ManipLLM/train
  
  bash finetune.sh

## Model Testing
The public code only infers on the final prompt without chain-of-thought, predicting the pose directly. 

It will first use the model to infer on all the test samples, and then interact with object in the simulator (SAPIEN).

Remember to add the checkpoints of [CLIP](https://drive.google.com/file/d/1XxfRxUL442Zh4NN0-JeIyNSJnLlTpxsu/view?usp=sharing), [LLaMa](same with the process in training), and [LLama_Adapter](https://drive.google.com/file/d/1JxxoLhV9lbS4iQNALU8vwUrRReGxS0eU/view?usp=sharing) under /ManipLLM/test/ckpts as well.

Place the [Manip](https://drive.google.com/file/d/1XxfRxUL442Zh4NN0-JeIyNSJnLlTpxsu/view?usp=sharing) checkpoints under /ManipLLM/test/exp or use your own trained checkpoint
    ```bash
    
    cd ./ManipLLM/test
    
    bash test.sh

## Acknowledgement
This repo benefits from [LLama_Adapter](https://github.com/OpenGVLab/LLaMA-Adapter) and [Where2act](https://github.com/daerduoCarey/where2act). Thanks for their wonderful works.
