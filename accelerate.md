# accelerate
accelerate launch \
    --config_file configs/fsdp_config.yaml \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$MACHINE_RANK \
    --num_processes 16 \
    --num_machines 2 \
    train.py \
    --model_name "meta-llama/Llama-2-70b-chat-hf" \
    --dataset_name "smangrul/code-chat-assistant-v1" \
    --max_seq_len 2048 \
    --max_steps 500 \
    --logging_steps 25 \
    --eval_steps 100 \
    --save_steps 250 \
    --bf16 True \
    --packing True \
    --output_dir "/shared_storage/sourab/experiments/full-finetune-llama-chat-asst" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --dataset_text_field "content" \
    --use_gradient_checkpointing True \
    --learning_rate 5e-5 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --use_flash_attn True
