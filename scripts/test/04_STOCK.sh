python3 -m run.test \
    --output results/04_STOCK.json \
    --model_id overfit-brothers/STOCK_SOLAR-10.7B-overfitting1 \
    --device cuda:0 \
    --custom_template skip_category  \
    --prompt_template NO
