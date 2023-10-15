date
#sleep 15000
#for dataset in cds electronic beauty
for dataset in cds electronic
do
{
  for model in new_din new_gru4rec new_sasrec
    do
    {
      for type in multi_metric_base_30
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

date
sleep 900
#for((i=1;i<=5;i++))
#do
#{
for dataset in cds electronic
do
{
  for model in new_din new_gru4rec new_sasrec
    do
    {
#      for type in base meta meta_attention
#      for type in base meta
      for type in multi_metric_base_finetune
  #    for type in base
  #    for type in meta meta_attention
      do
        {
          folder="${model}/${type}"
          echo ${folder}
          cd ${folder}
#          sh movielens_train.sh
          sh amazon_${dataset}_train.sh
          cd ../../
        } &
      done
    } &
    done
    wait
#    sleep 1500
#}
#done
#wait # 等待所有任务结束
#date
} &
done
wait # 等待所有任务结束
date
