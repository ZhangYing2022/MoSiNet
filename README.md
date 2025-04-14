## Code has been uploaded!  

This repository contains the code of **MoSiNet**.  

## Environment Setup  

1. **Install Python 3.8+**  
2. **Install dependencies**:  
   ```bash
   pip install -r requirements.txt

## Data Preparation
Place training data in /data.

## Training
```bash
python runner.py \
    --dataset_name=${DATASET_NAME} \
    --bert_name=${BERT_NAME} \
    --num_epochs=15 \
    --batch_size=16 \
    --lr=3e-5 \
    --warmup_ratio=0.06 \
    --eval_begin_epoch=1 \
    --seed=1234 \
    --do_train \
    --max_seq=80 \
    --use_prompt \
    --prompt_len=4 \
    --sample_ratio=1.0 \
    --save_path='ckpt/re/'

