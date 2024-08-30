python3 -m run.test \
    --output results/06_YI.json \
    --model_id overfit-brothers/Yi-Ko-34B-Chat-bnb-4bit-valdata-qlora \
    --device cuda:0 \
    --custom_template default \
    --prompt_template yi-ko \
    --load_in_4bit
