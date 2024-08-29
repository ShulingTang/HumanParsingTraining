# base
CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch --nproc_per_node 2 train_swin_cdg_mutil.py --arch swin_cdg \
--imagenet-pretrain ./pretrained/solider_swin_base.pth \
--data-dir /data0/tsl/data/full_19 \
--num-classes 19 \
--batch-size 3  \
--learning-rate 7e-3 \
--weight-decay 0 \
--optimizer sgd \
--syncbn \
--lr_divider 500 \
--cyclelr-divider 2 \
--warmup_epochs 30 \
--epochs 200 \
--schp-start 160 \
--cycle-epochs 10 \
--input-size 1024,768 \
--log-dir ./logs/swin_cdg_pretrained \
--model-restore ./logs/checkpoint_epoch.pth
