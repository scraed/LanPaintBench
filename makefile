.PHONY : train
run_imagenet:
	CUDA_VISIBLE_DEVICES=0 python run.py  --method LanPaint-5 --batch_size 16 --num_imgs 1000 --mask_type box checkerboard half outpaint --ite_step_size 0.15 --uld_alpha 0 --uld_friction 15 --param_m 1 --metrics --sample_steps 20 --suffix ""
	CUDA_VISIBLE_DEVICES=0 python run.py  --method LanPaint-10 --batch_size 16 --num_imgs 1000 --mask_type box checkerboard half outpaint --ite_step_size 0.15 --uld_alpha 0 --uld_friction 15 --param_m 1 --metrics --sample_steps 20 --suffix ""

