export CUDA_VISIBLE_DEVICES=1
cd ../../..
rm model.tar.gz
tar -czf model.tar.gz ./main_new ./loader ./model ./util ./module
cd scripts/new_gru4rec/meta

#ITERATION=30000
ITERATION=7100
#SNAPSHOT=10000
SNAPSHOT=710
ARCH_CONF_FILE=`pwd`/movielens_100k_conf.json
ARCH_CONFIG_CONTENT=`cat ${ARCH_CONF_FILE} | base64 -w 0`

GRADIENT_CLIP=5                     # !
BATCH_SIZE=1024

######################################################################
LEARNING_RATE=0.001
CHECKPOINT_PATH=lzq/SIGIR2023/checkpoint/movielens_100k_gru4rec_meta_new/
######################################################################

echo ${CHECKPOINT_PATH}
echo "Model save to ${CHECKPOINT_PATH}"


USER_DEFINED_ARGS="--model=meta_gru4rec --num_loading_workers=16 --arch_config=${ARCH_CONFIG_CONTENT} --learning_rate=${LEARNING_RATE} \
--max_gradient_norm=${GRADIENT_CLIP} --batch_size=${BATCH_SIZE} --snapshot=${SNAPSHOT} --max_steps=${ITERATION} --checkpoint_dir=${CHECKPOIN\
T_PATH}"

echo ${USER_DEFINED_ARGS}
/home/wf135777/data/odps_clt_release_64/bin/odpscmd -e "use graph_embedding_dev;pai \
-name pytorch180 -project algo_public \
-Dscript=\"file://`pwd`/../../../model.tar.gz\" \
-Dtables=\"odps://graph_embedding_intern_dev/tables/tds_movielens_100k_train_dataset_seq30\,odps://graph_embedding_intern_dev/tables/tds_movielens_100k_test_dataset_seq30\" \
-DentryFile=\"main_new/meta_train.py\" \
-Doversubscription=\"true\" \
-Dcluster='{\"worker\":{\"cpu\":100,\"memory\":16384,\"gpu\":100}}' \
-DuserDefinedParameters='${USER_DEFINED_ARGS}'\
"

echo "Training done: ${CHECKPOINT_PATH}"

