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

#          for file in movielens_ood_if_train movielens_ood_lof_train movielens_ood_ocsvm_train
          for file in movielens_ood_lof_train movielens_ood_ocsvm_train
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
#    sleep 2000
#    sleep 4200
#    sleep 6000
    sleep 10800
}
done
wait # 等待所有任务结束
date
