余弦退火
train：
python train_semi.py \
    --dataset ModelNet40 \
    --num_classes 40 \
    --n_labeled 800 \
    --batch_size 32 \
    --use_ma \
    --use_low_confidence_correction \
    --use_sharpening \
    --save ../checkpoints/ModelNet40_v1.2_KL/ ---更改

test：
python test_semi.py --dataset ModelNet40 --quantization_type swdc --n_labeled 800 --gpu_id 0 --save /home/new_disk_users/HOPE-fxed/checkpoints/ModelNet40_swdc/test_result/
