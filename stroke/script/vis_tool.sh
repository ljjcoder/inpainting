
python painter_gmcnn.py \
--load_model_dir /home/lijunjie/inpainting_gmcnn-master/tensorflow_MLD/tensorbord/gmcnn_paris_stroke_author/ \
--load_model_dir1 /home/lijunjie/inpainting_gmcnn-master/tensorflow_MLD/tensorbord/gmcnn_paris_stroke_mld/ \
--img_shapes 512,680 --mode silent

python painter_gmcnn.py \
--load_model_dir /home/lijunjie/inpainting_gmcnn-master/tensorflow_MLD/pretrain_model/places2_512x680_freeform/ \
--load_model_dir1 /home/lijunjie/inpainting_gmcnn-master/tensorflow_MLD/tensorbord/places2_stroke_MLD_bigmask/ \
--img_shapes 512,680 --mode silent

python xq_painter_gmcnn.py \
--load_model_dir /home/lijunjie/inpainting_gmcnn-master/tensorflow_MLD/pretrain_model/places2_512x680_freeform/ \
--load_model_dir1 /home/lijunjie/inpainting_gmcnn-master/tensorflow_MLD/tensorbord/places2_stroke_MLD_bigmask/ \
--img_shapes 512,680 --mode silent


python xq_painter_gmcnn.py \
--load_model_dir /home/lijunjie/inpainting_gmcnn-master/tensorflow_MLD/pretrain_model/places2_512x680_freeform/ \
--load_model_dir1 /home/lijunjie/inpainting_gmcnn-master/tensorflow_MLD/tensorbord/places2_stroke_MLD_bigmask/ \
--img_shapes 256,256 --mode silent
"/home/lijunjie/inpainting_gmcnn-master/tensorflow/tensorbord/places2-stroke-big-mask/"

python painter_gmcnn.py \
--load_model_dir /home/lijunjie/inpainting_gmcnn-master/tensorflow/tensorbord/places2-stroke-big-mask/ \
--load_model_dir1 /home/lijunjie/inpainting_gmcnn-master/tensorflow_MLD/tensorbord/places2_stroke_MLD_bigmask/ \
--img_shapes 256,256 --mode silent

python painter_ori_gmcnn.py \
--load_model_dir /home/lijunjie/inpainting_gmcnn-master/tensorflow_MLD/tensorbord/places2_stroke_MLD_bigmask/ \
--img_shapes 512,741 --mode silent


python painter_gmcnn.py \
--load_model_dir /home/lijunjie/inpainting_gmcnn-master/tensorflow_MLD/tensorbord/gmcnn_paris_stroke_author/ \
--load_model_dir1 /home/lijunjie/inpainting_gmcnn-master/tensorflow_MLD/tensorbord/gmcnn_paris_stroke_mld/ \
--img_shapes 256,256 --mode silent

python painter_gmcnn.py \
--load_model_dir /home/lijunjie/inpainting_gmcnn-master/tensorflow_MLD/tensorbord/gmcnn-HQ-bigstroke-authorcode/ \
--load_model_dir1 /home/lijunjie/inpainting_gmcnn-master/tensorflow_MLD/tensorbord/HQ-MLD-LW-stroke-bigmask_kernel8x8/ \
--img_shapes 256,256 --mode silent