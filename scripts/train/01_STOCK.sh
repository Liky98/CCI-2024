python3 -m run.train \
    --model_id kihoonlee/STOCK_SOLAR-10.7B \
    --save_local_dir_path resource/results/01_STOCK\
    --push_to_hub \
    --hub_model_name overfit-brothers/01_STOCK \
    --HF_TOKEN $HF_TOKEN \
    --custom_template default \
    --prompt_template kihoon_custom \
    --use_peft \
    --use_validation_data \
    --gradient_accumulation_steps 64 \
    --lr 5e-5 \
    --epoch 5 \
    --batch_size 1 \
    --max_seq_length 1024

