python train-wa-LORE.py --data-dir 'imagenet-data' \
    --log-dir 'trained_models' \
    --desc 'preresnet18_tiny-imagenet_lr0p2_Trades5_epoch100_bs512_fraction0p7_ls0p1_LORE_v1' \
    --data tiny-imagenet \
    --batch-size 512 \
    --batch-size-validation 128 \
    --model preact-resnet18\
    --num-adv-epochs 100 \
    --lr 0.2 \
    --beta 5.0 \
    --unsup-fraction 0.7 \
    --aux-data-filename '/home/xiangyuy/DM-Improves-AT/tinyimagenet_aug/tiny_edm_1m.npz' \
    --ls 0.1 \
    --LORE_type LORE_v1\
    --gpu_id 2

python train-wa-LORE.py --data-dir 'imagenet-data' \
    --log-dir 'trained_models' \
    --desc 'preresnet18_tiny-imagenet_lr0p2_MART_epoch100_bs512_fraction0p7_ls0p1_LORE_v1' \
    --data tiny-imagenet \
    --batch-size 512 \
    --batch-size-validation 128 \
    --model preact-resnet18\
    --num-adv-epochs 100 \
    --lr 0.2 \
    --mart \
    --beta 5.0 \
    --unsup-fraction 0.7 \
    --aux-data-filename '/home/xiangyuy/DM-Improves-AT/tinyimagenet_aug/tiny_edm_1m.npz' \
    --ls 0.1 \
    --LORE_type LORE_v1\
    --gpu_id 2

python train-wa-LORE.py --data-dir 'imagenet-data' \
    --log-dir 'trained_models' \
    --desc 'preresnet18_tiny-imagenet_lr0p2_TRADES5LSE_epoch100_bs512_fraction0p7_ls0p1_LORE_v1' \
    --data tiny-imagenet \
    --batch-size 512 \
    --batch-size-validation 128 \
    --model preact-resnet18\
    --num-adv-epochs 100 \
    --lr 0.2 \
    --LSE \
    --beta 5.0 \
    --unsup-fraction 0.7 \
    --aux-data-filename '/home/xiangyuy/DM-Improves-AT/tinyimagenet_aug/tiny_edm_1m.npz' \
    --ls 0.1 \
    --LORE_type LORE_v1\
    --gpu_id 2