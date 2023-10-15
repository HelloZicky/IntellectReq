date
#sleep 10800
for((i=1;i<=5;i++))
do
{
  for model in new_din new_gru4rec new_sasrec
    do
    {
#      for type in multi_metric_base_30 multi_metric_meta_30 multi_metric_meta_ood_uncertainty multi_metric_meta_ood_baseline multi_metric_meta_random_request
      for type in multi_metric_meta_30 multi_metric_meta_ood_uncertainty multi_metric_meta_ood_baseline multi_metric_meta_random_request
      do
        {
          folder="${model}/${type}"
          echo ${folder}
          cd ${folder}
#          if [ ${type} == "multi_metric_meta_ood_gru" ];then
##            for file in movielens_train movielens_ood_train movielens_ood2_train
#            for file in movielens_train movielens_ood_train  # 不要用ood2, ood2使用了target item
#            do
#              {
#                sh "${file}.sh"
#              }&
#             done
#
#          elif [ ${type} == "multi_metric_meta_random_request" ];then
          if [ ${type} == "multi_metric_meta_random_request" ];then
            sh movielens_train.sh

          elif [ ${type} == "multi_metric_meta_ood_baseline" ];then
            for file in movielens_ood_lof_train movielens_ood_ocsvm_train # 不要用ood2, ood2使用了target item
            do
              {
                sh "${file}.sh"
              }&
             done

          else
            sh movielens_ood_train.sh
          fi
          cd ../../
        } &
      done
    } &
    done
    wait
#    sleep 2000
#    sleep 4200
#    sleep 6000
    sleep 12000
}
done
wait # 等待所有任务结束
date
