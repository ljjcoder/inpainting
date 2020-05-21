python train.py --dataset places2 --data_file /home/lijunjie/generative_inpainting-master_global_local/data/filelist_places365-challenge/places2_chalenge_train_shuff.flist \
--mask_type stroke --gpu_ids 3 \
--load_model_dir ./pretrain_model/places2_512x680_freeform \
--pretrain_network 0 --batch_size 8 --tensorboard_folder ./tensorbord/places2_stroke_MLD_bigmask_retrain1/
