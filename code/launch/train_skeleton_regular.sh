python code/train_skeleton_regular.py \
    --train_data_list code/data/train_list.txt \
    --val_data_list code/data/val_list.txt \
    --data_root code/data \
    --model_name pct2 \
    --output_dir code/output/skeleton/any_pose "$@" \
    --random_pose 1 \
    --batch_size 16 \
    --epochs 2000 \
    --learning_rate 0.00001 \
    --num_sample 4096 \
    --vertices_sample 2048 \
    --pretrained_model checkpoints/skeleton/any_pose/best_model_1055.pkl \
# pretrained model can be set to continue training from "code/output/skeleton/any_pose/final_model.pkl"