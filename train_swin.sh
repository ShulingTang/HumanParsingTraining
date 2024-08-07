# tiny
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch --nproc_per_node 8 train.py --arch swin_cdg \
--imagenet-pretrain ./pretrained/solider_swin_base.pth \
--batch-size 8  \
--learning-rate 7e-3 \
--weight-decay 0 \
--optimizer sgd \
--syncbn \
--lr_divider 500 \
--cyclelr_divider 2 \
--warmup_epochs 30 \
--epochs 150 \
--schp-start 120 \
--input-size 512,512 \
--log-dir ./logs/mutil_gpu_swin_cdg \
