#python test_CA.py --dataset paris_streetview --data_file /home/lijunjie/generative_inpainting-master_global_local/data/places2_gt_1000/ \
--load_model_dir /home/lijunjie/inpainting_gmcnn-master/gmcc_code-leinao/tensorflow_mld/tensorbord/places2_rec_mld_lw_no_LL_globallocal_have_one_LL/ \
--random_mask 0 --use_CA_max 0
python test.py --dataset paris_streetview --data_file /home/lijunjie/generative_inpainting-master_global_local/data/places2_gt_1000/ \
--load_model_dir /home/lijunjie/inpainting_gmcnn-master/gmcc_code-leinao/tensorflow/tensorbord/places2_rec_dc/ \
--random_mask 0

python test.py --dataset paris_streetview --data_file /home/lijunjie/generative_inpainting-master_global_local/data/places2_gt_1000/ \
--load_model_dir /home/lijunjie/inpainting_gmcnn-master/gmcc_code-leinao/tensorflow_mld/tensorbord/places2_wave_loss_1.5_mld_0.001/ \
--random_mask 0