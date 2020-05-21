#python test.py --dataset paris_streetview --data_file /home/lijunjie/generative_inpainting-master_global_local/data/places2_gt_10000/ --load_model_dir /home/lijunjie/inpainting_gmcnn-master/tensorflow/tensorbord/places2-stroke-big-mask/ --mask_type stroke --random_mask 0 --img_shapes 256,256,3
#python test.py --dataset paris_streetview --data_file /home/lijunjie/generative_inpainting-master_global_local/data/places2_gt_10000/ --load_model_dir /home/lijunjie/inpainting_gmcnn-master/tensorflow/tensorbord/places2-stroke-big-mask/ --save_root2 /data4/lijunjie/gmcnn/authormodelretrain_bigmask_test_mask_90_002_10000 --mask_type stroke --random_mask 0 --img_shapes 256,256,3
#python test.py --dataset paris_streetview --data_file /home/lijunjie/generative_inpainting-master_global_local/data/places2_gt_1000/ --load_model_dir /home/lijunjie/inpainting_gmcnn-master/tensorflow/pretrain_model/places2_512x680_freeform/ --mask_type stroke --random_mask 0 --img_shapes 256,256,3
#python test.py --dataset paris_streetview --data_file /home/lijunjie/generative_inpainting-master_global_local/data/places2_gt_1000/ \
--load_model_dir /data4/lijunjie/tensorbord/places2_stroke_retrain_orimask/ \
--save_root2 /data4/lijunjie/gmcnn/places2_stroke_retrain_orimask//mask_90 \
--mask_type stroke --random_mask 0 --img_shapes 256,256,3 --mask_id 0

#python test.py --dataset paris_streetview --data_file /home/lijunjie/generative_inpainting-master_global_local/data/places2_gt_1000/ \
--load_model_dir /data4/lijunjie/tensorbord/places2_stroke_retrain/ \
--save_root2 /data4/lijunjie/gmcnn/places2_stroke_retrain_orimask/mask_90_002mask_big \
--mask_type stroke --random_mask 0 --img_shapes 256,256,3 --mask_id 1

python test.py --dataset paris_streetview --data_file /home/lijunjie/generative_inpainting-master_global_local/data/places2_gt_1000/ \
--load_model_dir /data4/lijunjie/tensorbord/places2_stroke_retrain/ \
--save_root2 /data4/lijunjie/gmcnn/places2_stroke_retrain_orimask/mask_60_stoke_172_big \
--mask_type stroke --random_mask 0 --img_shapes 256,256,3 --mask_id 2

#python test.py --dataset paris_streetview --data_file /home/lijunjie/generative_inpainting-master_global_local/data/places2_gt_1000/ \
--load_model_dir /data4/lijunjie/tensorbord/places2_stroke_retrain_orimask/ \
--save_root2 /data4/lijunjie/gmcnn/places2_stroke_retrain_orimask/mask_60 \
--mask_type stroke --random_mask 0 --img_shapes 256,256,3 --mask_id 3

#python test.py --dataset paris_streetview --data_file /home/lijunjie/generative_inpainting-master_global_local/data/places2_gt_1000/ \
--load_model_dir /data4/lijunjie/tensorbord/places2_stroke_retrain_orimask/ \
--save_root2 /data4/lijunjie/gmcnn/places2_stroke_retrain_orimask/mask_61 \
--mask_type stroke --random_mask 0 --img_shapes 256,256,3 --mask_id 4

#python test.py --dataset paris_streetview --data_file /home/lijunjie/generative_inpainting-master_global_local/data/places2_gt_1000/ \
--load_model_dir /data4/lijunjie/tensorbord/places2_stroke_retrain_orimask/ \
--save_root2 /data4/lijunjie/gmcnn/places2_stroke_retrain_orimask/mask_70 \
--mask_type stroke --random_mask 0 --img_shapes 256,256,3 --mask_id 5




#python test.py --dataset paris_streetview --data_file /home/lijunjie/generative_inpainting-master_global_local/data/places2_gt_10000/ \
--load_model_dir /data4/lijunjie/tensorbord/places2_stroke_retrain_orimask/ \
--save_root2 /data4/lijunjie/gmcnn/places2_stroke_retrain_orimask//mask_90_10000 \
--mask_type stroke --random_mask 0 --img_shapes 256,256,3 --mask_id 0

#python test.py --dataset paris_streetview --data_file /home/lijunjie/generative_inpainting-master_global_local/data/places2_gt_10000/ \
--load_model_dir /home/lijunjie/inpainting_gmcnn-master/tensorflow_MLD/tensorbord/places2_stroke_MLD_LW_bigmask/ \
--save_root2 /data4/lijunjie/gmcnn/places2_stroke_retrain_orimask/mask_90_002mask_10000_MLD \
--mask_type stroke --random_mask 0 --img_shapes 256,256,3 --mask_id 1

python test.py --dataset paris_streetview --data_file /home/lijunjie/generative_inpainting-master_global_local/data/places2_gt_10000/ \
--load_model_dir /data4/lijunjie/tensorbord/places2_stroke_retrain/ \
--save_root2 /data4/lijunjie/gmcnn/places2_stroke_retrain_orimask/mask_60_stoke_172_10000_big \
--mask_type stroke --random_mask 0 --img_shapes 256,256,3 --mask_id 2

#python test.py --dataset paris_streetview --data_file /home/lijunjie/generative_inpainting-master_global_local/data/places2_gt_10000/ \
--load_model_dir /data4/lijunjie/tensorbord/places2_stroke_retrain_orimask/ \
--save_root2 /data4/lijunjie/gmcnn/places2_stroke_retrain_orimask/mask_60_10000 \
--mask_type stroke --random_mask 0 --img_shapes 256,256,3 --mask_id 3

#python test.py --dataset paris_streetview --data_file /home/lijunjie/generative_inpainting-master_global_local/data/places2_gt_10000/ \
--load_model_dir /data4/lijunjie/tensorbord/places2_stroke_retrain_orimask/ \
--save_root2 /data4/lijunjie/gmcnn/places2_stroke_retrain_orimask/mask_61_10000 \
--mask_type stroke --random_mask 0 --img_shapes 256,256,3 --mask_id 4

#python test.py --dataset paris_streetview --data_file /home/lijunjie/generative_inpainting-master_global_local/data/places2_gt_10000/ \
--load_model_dir /data4/lijunjie/tensorbord/places2_stroke_retrain_orimask/ \
--save_root2 /data4/lijunjie/gmcnn/places2_stroke_retrain_orimask/mask_70_10000 \
--mask_type stroke --random_mask 0 --img_shapes 256,256,3 --mask_id 5