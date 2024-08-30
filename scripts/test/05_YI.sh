python3 -m run.test \
    --output results/05_YI.json \
    --model_id overfit-brothers/Yi-Ko-34B-Chat-bnb-4bit-valdata-adddata-qlora \
    --device cuda:0 \
    --custom_template default \
    --prompt_template yi-ko \
    --load_in_4bit

