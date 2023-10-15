date
#sleep 10800
#for(i=1;i<=5;i++]
#do
#{
for model in new_din new_gru4rec new_sasrec
do
{
#  for type in multi_metric_base_30 multi_metric_meta_30 multi_metric_meta_ood_uncertainty multi_metric_meta_ood_baseline multi_metric_meta_random_request multi_metric_base_finetune
  for type in multi_metric_base_30 multi_metric_meta_30 multi_metric_meta_ood_uncertainty multi_metric_meta_ood_baseline multi_metric_meta_random_request
  do
    {
      echo "${type}"
      folder="${model}/"${type}""
#      echo ${folder}
      cd ${folder}
      if (("${type}" == "multi_metric_meta_30" || "${type}" == "multi_metric_meta_ood_gru"));then
#         for file in amazon_beauty_train amazon_beauty_ood_train amazon_beauty_ood2_train
        for file in amazon_beauty_train amazon_beauty_ood_train  # 不要用ood2, ood2使用了target item
        do
          {
            sh "${file}.sh"
          }&
         done

      elif (("${type}" == "multi_metric_meta_random_request" || "${type}" == "multi_metric_base_finetune"));then
        sh "amazon_beauty_train.sh"

      elif (("${type}" == "multi_metric_meta_ood_baseline"));then
        for file in amazon_beauty_ood_lof_train amazon_beauty_ood_ocsvm_train # 不要用ood2, ood2使用了target item
        do
          {
            sh "${file}.sh"
          }&
         done

      else
        sh "amazon_beauty_ood_train.sh"
      fi
      cd ../../
    } &
  done
} &
done
wait
#    sleep 960
#    sleep 2100
#    sleep 4800
#}œ
#done
#wait # 等待所有任务结束
#date
