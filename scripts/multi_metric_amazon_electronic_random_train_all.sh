date
#sleep 10800
for((i=1;i<=5;i++))
do
{
  for model in new_din new_gru4rec new_sasrec
    do
    {
      for type in multi_metric_meta_random_request
      do
        {
          folder="${model}/${type}"
          echo ${folder}
          cd ${folder}
          sh amazon_electronic_train.sh
          cd ../../
        } &
      done
    } &
    done
    wait
    sleep 900
}
done
wait # 等待所有任务结束
date
