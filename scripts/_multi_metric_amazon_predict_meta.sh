date
#sleep 15000
for dataset in cds electronic beauty
do
{
  for model in new_din new_gru4rec new_sasrec
    do
    {
      for type in multi_metric_meta_30_pred
      do
        {
          folder="${model}/${type}"
          echo ${folder}
          cd ${folder}
          sh amazon_${dataset}_train.sh
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
