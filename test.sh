FPN_MODE='+v2'
NAME=v16_midv1_cim_bifpn_mda_mask
python3   dirl_test.py \
--dataset_root /media/sda/datasets/IHD \
--checkpoints_dir /media/sda/Harmonization/inharmonious/DIRLNet/${NAME} \
--batch_size 1 \
--gpu_ids 1 \
--input_nc 3 \
--output_nc 1 \
--resume 1 \
--is_train 0 \
--phase test \
--preprocess resize \
--no_flip \
--input_nc 3 \
--transition_type 'cim' \
--enriched_type 'mda' \
--decoder_type 'mid' \
--gate_ch -1 \
--ggd_ch 32 \
--cim_type 'bifpn' \
--backbone 'resnet34' \
--mda_mode 'mask' \
--mid_mode 'v1' \
--fpn_mode ${FPN_MODE} \
--resume -2


