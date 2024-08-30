python3 -m run.test \
    --output results/03_MISTRAL.json \
    --model_id overfit-brothers/Mistral-Nemo-Instruct-2407-bnb-4bit-valdata-qlora \
    --device cuda:0 \
    --custom_template default \
    --prompt_template no \
    --load_in_4bit
