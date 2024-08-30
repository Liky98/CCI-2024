python3 -m run.test \
    --output results/02_GEMMA.json \
    --model_id overfit-brothers/gemma-2-27b-it-bnb-4bit-valdata-qlora \
    --device cuda:0 \
    --custom_template default \
    --prompt_template no \
    --load_in_4bit

