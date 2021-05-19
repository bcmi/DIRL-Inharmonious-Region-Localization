MDA_MODE='mask'
LAMBDA_ATTENTION=1
NAME=dirl_mda_${MDA_MODE}_attnW_${LAMBDA_ATTENTION}_no_conv4

python3   dirl_test.py \
--dataset_root /media/sda/datasets/IHD \
--checkpoints_dir /media/sda/Harmonization/inharmonious/DIRL/${NAME} \
--batch_size 1 \
--gpu_ids 1 \
--input_nc 3 \
--output_nc 1 \
--is_train 0 \
--phase test \
--preprocess resize \
--no_flip \
--ggd_ch 32 \
--backbone 'resnet34' \
--mda_mode ${MDA_MODE} \
--resume -2


