wget http://archive.cloudera.com/cdh5/one-click-install/trusty/amd64/cdh5-repository_1.0_all.deb
dpkg -i cdh5-repository_1.0_all.deb

apt-get update
apt-get install vim tmux rsync htop --yes

apt-get install apt-transport-https

apt-get update
apt-get install hadoop-hdfs-fuse --yes --allow-unauthenticated


mkdir /workspace
hadoop-fuse-dfs $PAI_DEFAULT_FS_URI /workspace

# Above are default settings

pip install tqdm
model=transformer
PROBLEM=IWSLT14_DEEN
SETTING=transformer_small

ls /workspace

prefix_folder=/workspace/v-lijwu/transformer_pytorch_learn_loss
MODEL_DIR=${prefix_folder}/$model/$PROBLEM/$SETTING
mkdir -p ${MODEL_DIR}

CUDA_VISIBLE_DEVICES=0 python ${prefix_folder}/train.py ${prefix_folder}/data/data-bin/iwslt14.tokenized.de-en \
	--clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
	--arch $SETTING --save-dir ${MODEL_DIR} \
	--teacher-save-dir ${prefix_folder}/${model}/${PROBLEM}/teacher_dir \
	--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
	--lr-scheduler inverse_sqrt --lr 0.25 --optimizer nag --warmup-init-lr 0.25 \
	--warmup-updates 4000 --max-update 100000  --position timing --no-progress-bar \
	--max-pretrain-update 60000 --meta-iter 200 --max-meta-update 5000 --loss-lr 0.010 


