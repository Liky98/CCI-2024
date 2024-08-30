python3 -m run.train \
    --model_id kihoonlee/STOCK_SOLAR-10.7B \
    --save_local_dir_path resource/results/04_STOCK\
    --push_to_hub \
    --hub_model_name overfit-brothers/04_STOCK \
    --HF_TOKEN $HF_TOKEN \
    --custom_template skip_category  \
    --prompt_template NO \
    --use_peft \
    --use_validation_data \
    --gradient_accumulation_steps 1 \
    --lr 2e-5 \
    --epoch 1 \
    --batch_size 8 \
    --max_seq_length 1024

