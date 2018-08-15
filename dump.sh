nvidia-smi

pwd
REMOTE_CODE_PATH=/hdfs/sdrgvc/v-lijwu/bilm-tf-master

export PYTHONPATH=$REMOTE_CODE_PATH:$PYTHONPATH
# sudo pip install h5py

# export CUDA_VISIBLE_DEVICES=0,1,2
# export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 $REMOTE_CODE_PATH/bin/dump_weights.py \
    --save_dir=$REMOTE_CODE_PATH/model \
    --outfile=$REMOTE_CODE_PATH/model/weight.hdf5
