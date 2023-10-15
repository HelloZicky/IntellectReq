date
#sleep 15000
#for dataset in cds electronic beauty
#for dataset in cds electronic
for dataset in alipay
do
{
  for model in new_din new_gru4rec new_sasrec
    do
    {
      for type in multi_metric_base_30_apg multi_metric_meta_random_request_apg multi_metric_meta_ood_baseline_apg multi_metric_meta_30_apg multi_metric_meta_ood_uncertainty5_apg
#      for type in multi_metric_base_30 multi_metric_meta_30 multi_metric_meta_random_request multi_metric_meta_ood_uncertainty5
      do
        {
          folder="${model}/${type}"
          echo ${folder}
          cd ${folder}
          if [ ${type} = "multi_metric_meta_30" ];then
            for file in ${dataset}_train ${dataset}_ood_train  # 不要用ood2, ood2使用了target item
            do
              {
                sh "${file}.sh"
              }&
             done
          elif [ ${type} = "multi_metric_meta_random_request" ];then
            sh ${dataset}_train.sh
          elif [ ${type} = "multi_metric_meta_ood_baseline" ];then
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
