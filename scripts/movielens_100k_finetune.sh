date
#sleep 10800
for((i=1;i<=5;i++))
do
{
  for model in new_din new_gru4rec new_sasrec
    do
    {
#      for type in base meta meta_attention
#      for type in base meta
      for type in base_finetune
  #    for type in base
  #    for type in meta meta_attention
      do
        {
          folder="${model}/${type}"
          echo ${folder}
          cd ${folder}
          sh movielens_100k_train.sh
          cd ../../
        } &
      done
    } &
    done
    wait
    sleep 3600
}
done
wait # 等待所有任务结束
date
