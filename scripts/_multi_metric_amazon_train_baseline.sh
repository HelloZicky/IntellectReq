date
#sleep 15000
for dataset in cds electronic
do
{
  for model in new_din new_gru4rec new_sasrec
#  for model in new_sasrec
    do
    {
      for type in multi_metric_meta_random_request multi_metric_meta_ood_baseline
#      for type in multi_metric_meta_random_request
      do
        {
          folder="${model}/${type}"
          echo ${folder}
          cd ${folder}
          if [ ${type} = "multi_metric_meta_random_request" ];then
            sh amazon_${dataset}_train.sh
          else
            for file in amazon_${dataset}_ood_lof_train amazon_${dataset}_ood_ocsvm_train
            do
              {
                sh "${file}.sh"
              } &
            done
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
