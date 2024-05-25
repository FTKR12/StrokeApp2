python test.py \
    --dataroot /mnt/strokeapp/Datasets/merged_ctmri \
    --name pix2pix \
    --model pix2pix \
    --netG unet_256 \
    --direction AtoB \
    --dataset_mode aligned \
    --norm batch \
    --gpu_ids 0

