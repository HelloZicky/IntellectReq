date
#sleep 15000
#for dataset in cds electronic beauty
#for dataset in cds electronic
#for dataset in cds
#for dataset in electronic
for dataset in book music
#for dataset in book
#for dataset in alipay
do
{
  for model in new_din new_gru4rec new_sasrec
#  for model in new_din
    do
    {
#      for type in multi_metric_base_30 multi_metric_meta_random_request multi_metric_meta_ood_baseline multi_metric_meta_30 multi_metric_meta_ood_uncertainty5
#      for type in multi_metric_base_30 multi_metric_meta_30 multi_metric_meta_random_request multi_metric_meta_ood_baseline multi_metric_meta_ood_uncertainty5
#      for type in multi_metric_meta_random_request multi_metric_meta_ood_baseline
#      for type in multi_metric_meta_30_apg multi_metric_meta_random_request_apg multi_metric_meta_ood_baseline_apg multi_metric_meta_ood_uncertainty5_apg
#      for type in multi_metric_meta_30_apg multi_metric_meta_random_request_apg multi_metric_meta_ood_baseline_apg multi_metric_meta_ood_uncertainty5_apg
#      for type in multi_metric_meta_random_request_apg multi_metric_meta_ood_uncertainty5_apg
#      for type in multi_metric_meta_ood_uncertainty5_apg
      for type in multi_metric_meta_30_apg
#      for type in multi_metric_meta_ood_baseline_apg
      do
        {
          folder="${model}/${type}"
          echo ${folder}
          cd ${folder}
          if [ ${type} = "multi_metric_meta_30_apg" ];then
#            for file in douban_${dataset}_train douban_${dataset}_ood_train  # 不要用ood2, ood2使用了target item
            for file in douban_${dataset}_train  # 不要用ood2, ood2使用了target item
#            for file in douban_${dataset}_ood_train  # 不要用ood2, ood2使用了target item
            do
              {
                sh "${file}.sh"
              }&
             done
          elif [ ${type} = "multi_metric_meta_random_request_apg" ];then
            sh douban_${dataset}_train.sh
          elif [ ${type} = "multi_metric_meta_ood_baseline_apg" ];then
#            for file in douban_${dataset}_ood_lof_train douban_${dataset}_ood_ocsvm_train
            for file in douban_${dataset}_ood_ocsvm_train
            do
              {
                sh "${file}.sh"
              } &
            done
          else
            sh douban_${dataset}_ood_train.sh
#            echo douban_${dataset}_ood_train.sh
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
