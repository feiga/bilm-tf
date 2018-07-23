REMOTE_CODE_PATH=/hdfs/sdrgvc/v-lijwu/bilm-tf-master
DATA_PATH=$REMOTE_CODE_PATH/data

wget http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz -P $DATA_PATH
tar zxvf $DATA_PATH/1-billion-word-language-modeling-benchmark-r13output.tar.gz -C $DATA_PATH/
