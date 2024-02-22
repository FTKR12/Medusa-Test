CUDA_VISIBLE_DEVICES=0 python -m medusa.inference.cli \
    --model FasterDecoding/medusa-1.0-zephyr-7b-beta \
    --load-in-4bit