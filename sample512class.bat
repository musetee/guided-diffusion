@echo off

set SAMPLE_FLAGS=--batch_size 1 --num_samples 2 --timestep_respacing 250
set MODEL_FLAGS=--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 512 --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --learn_sigma True

call F:\yang_Environments\torch\venv\Scripts\activate

python classifier_sample.py --classifier_scale 0.5 --classifier_path ckpt/512x512_classifier.pt --model_path ckpt/512x512_diffusion.pt %SAMPLE_FLAGS% %MODEL_FLAGS%

pause
