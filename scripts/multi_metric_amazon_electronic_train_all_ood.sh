date
#sleep 15000
for((i=1;i<=5;i++))
do
{
  for model in new_din new_gru4rec new_sasrec
    do
    {
      for type in multi_metric_meta_30 multi_metric_meta_ood_uncertainty
      do
        {
          folder="${model}/${type}"
          echo ${folder}
          cd ${folder}

          sh amazon_electronic_ood_train.sh
          
          cd ../../
        } &
      done
    } &
    done
    wait
#    sleep 960
    sleep 2100
#    sleep 4200
}
done
wait # 等待所有任务结束
date
