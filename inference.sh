CUDA_VISIBLE_DEVICES=0 python -m medusa.inference.cli \
    --model weight_medusaheads_medusa_mlp_ELYZA-japanese-Llama-2-7b_medusa_3_lr_0.001_layers_1 \
    --conv-system-msg "あなたは誠実で優秀な日本人のアシスタントです。" \
    --max-steps 256