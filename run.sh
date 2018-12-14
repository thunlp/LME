export CUDA_VISIBLE_DEVICES=2

# for lamb in 0.001 0.005 0.01; do
#     python -u train.py gillick attentive --feature --hier --test --lamb $lamb
#     echo $lamb
#     echo
#     echo
# done

python -u train.py gillick attentive --feature --hier --test --lamb 0.001