mda_mode='mask_gb'
LAMBDA_ATTENTION=1
NAME=v16_mda_${mda_mode}_attnW_${LAMBDA_ATTENTION}

python3  dirl_train.py \
--dataset_root /media/sda/datasets/IHD \
--checkpoints_dir /media/sda/Harmonization/inharmonious/DIRLNet/${NAME} \
--batch_size 2 \
--gpu_ids 0 \
--preprocess resize_and_crop \
--save_epoch_freq 5 \
--is_train 1 \
--lr 1e-4 \
--nepochs 60 \
--ggd_ch 32 \
--backbone 'resnet34' \
--mda_mode ${mda_mode} \
--loss_mode '' \
--lambda_attention ${LAMBDA_ATTENTION} 
# --resume 10
