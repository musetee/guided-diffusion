@echo off

set SAMPLE_FLAGS=--batch_size 1 --num_samples 2 --timestep_respacing 250
set MODEL_FLAGS=--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 128 --learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True

call F:\yang_Environments\torch\venv\Scripts\activate

python image_sample.py --model_path ckpt/128x128_diffusion.pt %MODEL_FLAGS% %SAMPLE_FLAGS%

pause