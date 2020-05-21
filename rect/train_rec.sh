 python train_test.py --dataset places2 --data_file /home/lijunjie/generative_inpainting-master_global_local/data/filelist_places365-challenge/places2_chalenge_train_shuff.flist \
--mask_type rect --gpu_ids 3 --load_model_dir /home/lijunjie/inpainting_gmcnn-master/tensorflow_MLD/pretrain_model/places2_512x680_freeform \
--tensorboard_folder /data4/lijunjie/tensorbord/places2_rec_only_MLD_loss_correct_test_retrain2/ --pretrain_network 0 --batch_size 8  --max_iters 80000 \
--viz_steps 400 --train_spe 10000
 #python train_test.py \
 --dataset places2 \
 --data_file /home/lijunjie/generative_inpainting-master_global_local/data/train_paris_street_name.txt \
 --mask_type rect \
 --gpu_ids 1 \
 --load_model_dir  /home/lijunjie/inpainting_gmcnn-master/tensorflow_MLD/tensorbord/paris-streetview_256x256_rect/paris-streetview_256x256_rect/ \
 --tensorboard_folder ./tensorbord/paris_street_rec_mld_dc/ \
 --pretrain_network 0 --batch_size 8  --max_iters 90002 --viz_steps 2000 --train_spe 10000