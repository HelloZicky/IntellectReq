date
#sleep 15000
#for dataset in cds electronic beauty
#for dataset in cds electronic
for dataset in book music
#for dataset in alipay
do
{
  for model in new_din new_gru4rec new_sasrec
    do
    {
#      for type in multi_metric_base_30 multi_metric_meta_random_request multi_metric_meta_ood_baseline multi_metric_meta_30 multi_metric_meta_ood_uncertainty5
#      for type in multi_metric_base_30 multi_metric_meta_30 multi_metric_meta_random_request multi_metric_meta_ood_baseline multi_metric_meta_ood_uncertainty5
#      for type in multi_metric_meta_random_request multi_metric_meta_ood_baseline
      for type in multi_metric_meta_30 multi_metric_meta_random_request multi_metric_meta_ood_baseline multi_metric_meta_ood_uncertainty5
      do
        {
          folder="${model}/${type}"
          echo ${folder}
          cd ${folder}
          if [ ${type} = "multi_metric_meta_30" ];then
            for file in douban_${dataset}_train douban_${dataset}_ood_train  # 不要用ood2, ood2使用了target item
#            for file in douban_${dataset}_ood_train  # 不要用ood2, ood2使用了target item
            do
              {
                sh "${file}.sh"
              }&
             done
          elif [ ${type} = "multi_metric_meta_random_request" ];then
            sh douban_${dataset}_train.sh
          elif [ ${type} = "multi_metric_meta_ood_baseline" ];then
            for file in douban_${dataset}_ood_lof_train douban_${dataset}_ood_ocsvm_train
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
