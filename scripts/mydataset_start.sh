# CUDA_VISIBLE_DEVICES=0 python3 -u scripts/mydataset_start.py \
#     --output_dir log/mydataset_start \
#     --data_root mydataset \
#     --model_size base \
#     --load_path checkpoints/checkpoint-4frames.pth \
#     --num_frames 4 \
#     --stride 2 \
#     --window_step 1 \
#     --model anytouch \
#     --sensors digit gelsight

# Exact step-14 frame extraction. The same frame is repeated to match the 4-frame model input.
CUDA_VISIBLE_DEVICES=0 python3 -u scripts/mydataset_start.py \
    --output_dir log/mydataset_start_frame14 \
    --data_root mydataset \
    --model_size base \
    --load_path checkpoints/checkpoint-4frames.pth \
    --num_frames 4 \
    --stride 2 \
    --window_step 1 \
    --model anytouch \
    --sensors digit gelsight \
    --single_step_mode \
    --single_step_value 14 \
    --single_step_movements center right left down up \
    --single_frame_mode
