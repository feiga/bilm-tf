nvidia-smi

pwd
REMOTE_CODE_PATH=/hdfs/sdrgvc/v-lijwu/bilm-tf-master
DATA_PATH=$REMOTE_CODE_PATH/data

export PYTHONPATH=$REMOTE_CODE_PATH:$PYTHONPATH
# sudo pip install h5py

PERMUTE_NUMBER=6
DIM=2048  # make it to be half of the 4 multidirectional model
PROJECT_DIM=512  # make it to be half of the 4 multidirectional model
MODEL_DIR=$REMOTE_CODE_PATH/model_omni_${PERMUTE_NUMBER}_${DIM}_${PROJECT_DIM}
mkdir -p $MODEL_DIR

# export CUDA_VISIBLE_DEVICES=0,1,2
# export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 $REMOTE_CODE_PATH/bin/train_omni_elmo.py \
    --train_prefix=$DATA_PATH/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/* \
    --vocab_file=$DATA_PATH/vocab-2016-09-10.txt \
    --save_dir=$MODEL_DIR \
    --permute_number=$PERMUTE_NUMBER \
    --dim=$DIM \
    --projection_dim=$PROJECT_DIM
