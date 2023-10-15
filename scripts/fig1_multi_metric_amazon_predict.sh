date
#sleep 15000
#for dataset in cds electronic beauty
for dataset in cds electronic
do
{
  for model in new_din new_gru4rec new_sasrec
    do
    {
      for type in fig1_multi_metric_meta_30_pred fig1_multi_metric_meta_ood_uncertainty_pred5
      do
        {
          folder="${model}/${type}"
          echo ${folder}
          cd ${folder}
          sh amazon_${dataset}_ood_train.sh
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
