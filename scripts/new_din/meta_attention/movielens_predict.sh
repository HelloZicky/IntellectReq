export CUDA_VISIBLE_DEVICES=1
cd ../../..
rm model.tar.gz
tar -czf model.tar.gz ./main ./loader ./model ./util ./module
cd scripts/DIN/meta

######################################################################
MODEL_NAME=movielens_din_meta
#STEP=20000
#STEP=8000
#STEP=16000
#STEP=24000
#STEP=32000
#STEP=40000
#STEP=48000
#STEP=56000
#STEP=64000
#STEP=72000
STEP=80000
####################################################################

MODEL_PATH=lzq/SIGIR2023/checkpoint/${MODEL_NAME}
OUTPUT_VERSION=${MODEL_NAME}_${STEP}

PROJECT=graph_embedding_dev
ROLEARN=acs:ram::1374191468338678:role/wind-graph

/home/wf135777/data/odps_clt_release_64/bin/odpscmd -e "use graph_embedding_dev;pai \
-name pytorch180 -project algo_public \
-Dscript=\"file://`pwd`/../../../model.tar.gz\" \
-Dtables=\"odps://graph_embedding_intern_dev/tables/tds_movielens_train_dataset_seq30\" \
-Doutputs=\"odps://graph_embedding_intern_dev/tables/tds_movielens_rerank_model_infer_result_seq30_meta/version=torch_${OUTPUT_VERSION}\"
-DentryFile=\"main/meta_predict.py\" \
-Doversubscription=\"true\" \
-DworkerCount=\"8\" \
-DenableDynamicCluster=\"true\"\
-Dcluster='{\"worker\":{\"cpu\":800,\"memory\":8192,\"gpu\":100}}' \
-DuserDefinedParameters=\"--batch_size=128 --num_loading_workers=8 --checkpoint_dir='$MODEL_PATH' --step=${STEP} \" \
"

echo "Training done: ${CHECKPOINT_PATH}"
