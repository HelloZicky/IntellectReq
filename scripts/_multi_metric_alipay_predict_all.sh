date
#sleep 15000
for dataset in alipay
do
{
  for model in new_din new_gru4rec new_sasrec
    do
    {
      for type in multi_metric_meta_30_pred multi_metric_meta_ood_uncertainty_pred5 multi_metric_meta_random_request_pred multi_metric_meta_ood_baseline_pred
#      for type in multi_metric_meta_random_request_pred multi_metric_meta_ood_baseline_pred
#      for type in multi_metric_meta_ood_uncertainty_pred5 multi_metric_meta_random_request_pred
#      for type in multi_metric_meta_random_request_pred
#      for type in multi_metric_meta_ood_baseline_pred
      do
        {
          folder="${model}/${type}"
          echo ${folder}
          cd ${folder}
          if [ ${type} = "multi_metric_meta_random_request_pred" ];then
            sh ${dataset}_train.sh
#            echo ${dataset}_train.sh
          elif [ ${type} = "multi_metric_meta_ood_baseline_pred" ];then
            for file in ${dataset}_ood_lof_train ${dataset}_ood_ocsvm_train
            do
              {
                sh "${file}.sh"
              } &
            done
          else
            sh ${dataset}_ood_train.sh
#            echo ${dataset}_ood_train.sh
          fi
          cd ../../
        } &
      done
    } &
    done
    wait
#    sleep 960
#    sleep 360
#    sleep 4200
}
done
wait # 等待所有任务结束
date
