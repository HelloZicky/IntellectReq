date
#sleep 10800
for((i=1;i<=5;i++))
do
{
  for model in new_din new_gru4rec new_sasrec
    do
    {
      for type in multi_metric_meta_ood_baseline
      do
        {
          folder="${model}/${type}"
          echo ${folder}
          cd ${folder}

#          for file in amazon_beauty_ood_if_train amazon_beauty_ood_lof_train amazon_beauty_ood_ocsvm_train
          for file in amazon_beauty_ood_lof_train amazon_beauty_ood_ocsvm_train
          do
            {
              sh "${file}.sh"
            }&
          done

          cd ../../
        } &
      done
    } &
    done
    wait
#    sleep 300
#    sleep 480
    sleep 1500
}
done
wait # 等待所有任务结束
date
