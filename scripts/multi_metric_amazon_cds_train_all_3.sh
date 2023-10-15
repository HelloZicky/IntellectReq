date
#sleep 10800
#for(i=1;i<=5;i++]
#do
#{
for model in new_din new_gru4rec new_sasrec
do
{
#  for type in multi_metric_base_30 multi_metric_meta_30 multi_metric_meta_ood_uncertainty multi_metric_meta_ood_baseline multi_metric_meta_random_request multi_metric_base_finetune
#  for type in multi_metric_meta_30 multi_metric_meta_ood_baseline multi_metric_base_finetune
  for type in multi_metric_meta_ood_baseline
#  for type in multi_metric_meta_ood_uncertainty multi_metric_meta_ood_baseline multi_metric_meta_random_request
#  for type in multi_metric_base_30 multi_metric_meta_30 multi_metric_meta_ood_uncertainty multi_metric_meta_ood_baseline multi_metric_meta_random_request
  do
    {
      echo "${type}"
      folder="${model}/"${type}""
#      echo ${folder}
      cd ${folder}
      if [ ${type} = "multi_metric_meta_30" ]; then
#         for file in amazon_cds_train amazon_cds_ood_train amazon_cds_ood2_train
        sh "amazon_cds_ood_train.sh"
#      fi

      elif [ ${type} = "multi_metric_base_finetune" ]; then
        sh "amazon_cds_train.sh"
#      fi

      elif [ ${type} = "multi_metric_meta_ood_baseline" ]; then
        sh "amazon_cds_ood_lof_train.sh"
#      fi

#      if [ ${type}=="multi_metric_base_30" ] || [ ${type}=="multi_metric_base_finetune" ] || [ ${type}=="multi_metric_meta_ood_uncertainty" ]; then
      else
        sh "amazon_cds_ood_train.sh"
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
