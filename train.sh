mda_mode='mask'
LAMBDA_ATTENTION=1
NAME=dirl_mda_${mda_mode}_attnW_${LAMBDA_ATTENTION}

nohup python3 -u  dirl_train.py \
--dataset_root /media/sda/datasets/IHD \
--checkpoints_dir /media/sda/Harmonization/inharmonious/DIRLNet/${NAME} \
--batch_size 8 \
--gpu_ids 1 \
--preprocess resize_and_crop \
--save_epoch_freq 5 \
--is_train 1 \
--lr 1e-4 \
--nepochs 60 \
--ggd_ch 32 \
--backbone 'resnet34' \
--mda_mode ${mda_mode} \
--loss_mode '' \
--lambda_attention 1 \
> ${NAME}.log 2>&1 & 

# --resume 38 \