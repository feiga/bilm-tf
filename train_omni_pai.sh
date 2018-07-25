wget http://archive.cloudera.com/cdh5/one-click-install/trusty/amd64/cdh5-repository_1.0_all.deb
dpkg -i cdh5-repository_1.0_all.deb

apt-get update
apt-get install vim tmux rsync htop --yes

apt-get install apt-transport-https

apt-get update
apt-get install hadoop-hdfs-fuse --yes --allow-unauthenticated


mkdir /workspace
hadoop-fuse-dfs $PAI_DEFAULT_FS_URI /workspace]

CODE_PATH=/workspace/v-lijwu/bilm-tf-master
DATA_PATH=$CODE_PATH/data

export PYTHONPATH=$CODE_PATH:$PYTHONPATH
# sudo pip install h5py

# export CUDA_VISIBLE_DEVICES=0,1,2
# export CUDA_VISIBLE_DEVICES=0,1,2,3
python $CODE_PATH/bin/train_elmo_omni.py \
    --train_prefix=$DATA_PATH/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/* \
    --vocab_file=$DATA_PATH/vocab-2016-09-10.txt \
    --save_dir=$CODE_PATH/model_omni
