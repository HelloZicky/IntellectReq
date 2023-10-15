export CUDA_VISIBLE_DEVICES=1
cd ../../..
rm model.tar.gz
tar -czf model.tar.gz ./main ./loader ./model ./util ./module
cd scripts/DIN/base

######################################################################
MODEL_NAME=alipay_din_base
#STEP=4100
STEP=8200
#STEP=12300
#STEP=16400
#STEP=20500
#STEP=24600
#STEP=28700
#STEP=32800
#STEP=36900
#STEP=41000
####################################################################

MODEL_PATH=lzq/SIGIR2023/checkpoint/${MODEL_NAME}
OUTPUT_VERSION=${MODEL_NAME}_${STEP}

PROJECT=graph_embedding_dev
ROLEARN=acs:ram::1374191468338678:role/wind-graph

# tunnel upload -fd ';' /mnt4/lzq/planetary/from141/dataset/Alipay/final/new_71_encode/alipay_overall71_train.txt tds_alipay_train_dataset;

/home/wf135777/data/odps_clt_release_64/bin/odpscmd -e "use graph_embedding_dev;pai \
-name pytorch180 -project algo_public \
-Dscript=\"file://`pwd`/../../../model_alipay.tar.gz\" \
-Dtables=\"odps://graph_embedding_intern_dev/tables/tds_alipay_test_dataset_seq30\" \
-Doutputs=\"odps://graph_embedding_intern_dev/tables/tds_alipay_rerank_model_infer_result_seq30/version=torch_${OUTPUT_VERSION}\"
-DentryFile=\"main/vanilla_predict.py\" \
-Doversubscription=\"true\" \
-DworkerCount=\"8\" \
-DenableDynamicCluster=\"true\"\
-Dcluster='{\"worker\":{\"cpu\":800,\"memory\":8192,\"gpu\":100}}' \
-DuserDefinedParameters=\"--batch_size=128 --num_loading_workers=8 --checkpoint_dir='$MODEL_PATH' --step=${STEP} \" \
"

echo "Training done: ${CHECKPOINT_PATH}"
