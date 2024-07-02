python train-wa-LOAT.py --data-dir 'tiny-imagenet-data'\
    --log-dir 'trained_models' \
    --desc 'prn18_tiny-imagenet_lr0p2_Trades5_epoch100_bs512_fraction0p7_ls0p1_LORE' \
    --data tiny-imagenets \
    --batch-size 512 \
    --batch-size-validation 128 \
    --model preact-resnet18\
    --num-adv-epochs 100 \
    --lr 0.2 \
    --beta 5.0 \
    --unsup-fraction 0.7 \
    --aux-data-filename '~/DM-Improves-AT/tinyimagenet_aug/tiny_edm_1m.npz' \
    --ls 0.1 \
    --gpu_id 0 \
    --LORE_type LORE

python train-wa-LOAT.py --data-dir 'tiny-imagenet-data'\
    --log-dir 'trained_models' \
    --desc 'prn18_tiny-imagenet_lr0p2_Trades5LSE_epoch100_bs512_fraction0p7_ls0p1' \
    --data tiny-imagenets \
    --batch-size 512 \
    --batch-size-validation 128 \
    --model preact-resnet18\
    --num-adv-epochs 100 \
    --LSE \
    --lr 0.2 \
    --beta 5.0 \
    --unsup-fraction 0.7 \
    --aux-data-filename '~/DM-Improves-AT/tinyimagenet_aug/tiny_edm_1m.npz' \
    --ls 0.1 \
    --gpu_id 0 \
    --LORE_type None

python train-wa-LOAT.py --data-dir 'tiny-imagenet-data'\
    --log-dir 'trained_models' \
    --desc 'prn18_tiny-imagenet_lr0p2_Trades5LSE_epoch100_bs512_fraction0p7_ls0p1_LORE_v1' \
    --data tiny-imagenets \
    --batch-size 512 \
    --batch-size-validation 128 \
    --model preact-resnet18\
    --num-adv-epochs 100 \
    --LSE \
    --lr 0.2 \
    --beta 5.0 \
    --unsup-fraction 0.7 \
    --aux-data-filename '~/DM-Improves-AT/tinyimagenet_aug/tiny_edm_1m.npz' \
    --ls 0.1 \
    --gpu_id 0 \
    --LORE_type LORE_v1

python train-wa-LOAT.py --data-dir 'tiny-imagenet-data'\
    --log-dir 'trained_models' \
    --desc 'prn18_tiny-imagenet_lr0p2_Trades5LSE_epoch100_bs512_fraction0p7_ls0p_LORE' \
    --data tiny-imagenets \
    --batch-size 512 \
    --batch-size-validation 128 \
    --model preact-resnet18\
    --num-adv-epochs 100 \
    --LSE \
    --lr 0.2 \
    --beta 5.0 \
    --unsup-fraction 0.7 \
    --aux-data-filename '~/DM-Improves-AT/tinyimagenet_aug/tiny_edm_1m.npz' \
    --ls 0.1 \
    --gpu_id 0 \
    --LORE_type LORE
