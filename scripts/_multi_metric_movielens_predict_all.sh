date
#sleep 15000
#for dataset in cds electronic beauty
for dataset in movielens
do
{
  for model in new_din new_gru4rec new_sasrec
    do
    {
      for type in multi_metric_meta_30_pred multi_metric_meta_ood_uncertainty_pred multi_metric_meta_random_request_pred
      do
        {
          folder="${model}/${type}"
          echo ${folder}
          cd ${folder}
          if [ ${type} == "multi_metric_meta_random_request_pred" ];then
#            sh amazon_${dataset}_train.sh
            sh movielens_train.sh
#            echo amazon_${dataset}_train.sh
          else
#            sh amazon_${dataset}_ood_train.sh
            sh movielens_ood_train.sh
#            echo amazon_${dataset}_ood_train.sh
          fi
          cd ../../
        } &
      done
    } &
    done
    wait
#    sleep 960
    sleep 360
#    sleep 4200
}
done
wait # 等待所有任务结束
date
