accelerate launch controlnet_train.py \
 --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
 --output_dir="controlnet-landmark" \
 --dataset_name=saeu5407/celebahq_de_identification \
 --conditioning_image_column=landmark \
 --image_column=default \
 --caption_column=text \
 --resolution=256 \
 --mixed_precision="fp16" \
 --learning_rate=1e-5 \
<<<<<<< HEAD
 --validation_image "../../datasets/test/sample_landmark.png" "../../datasets/test/sample2_landmark.png"\
 --validation_prompt "a men in cafe" "a middle-aged black rapper in a black hat" \
 --train_batch_size=50 \
 --num_train_epochs=100 \
=======
 --validation_image "../../datasets/test/sample_landmark.png" \
 --validation_prompt "" \
 --train_batch_size=4 \
 --num_train_epochs=50 \
>>>>>>> 538119184c8def9716817786db97f8f55fefdde7
 --tracker_project_name="controlnet-landmark" \
 --enable_xformers_memory_efficient_attention \
 --checkpointing_steps=20000 \
 --checkpoints_total_limit=3 \
 --validation_steps=20000 \
 --use_8bit_adam \
 --gradient_checkpointing \
 --enable_xformers_memory_efficient_attention \
 --set_grads_to_none \
 --gradient_accumulation_steps=4 \
 --report_to wandb \
 --push_to_hub