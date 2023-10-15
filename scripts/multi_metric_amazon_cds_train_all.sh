date
#sleep 10800
for((i=1;i<=5;i++))
do
{
  for model in new_din new_gru4rec new_sasrec
    do
    {
      for type in multi_metric_base_30 multi_metric_meta_30 multi_metric_meta_ood_gru
      do
        {
          folder="${model}/${type}"
          echo ${folder}
          cd ${folder}
          if [ ${type} == "multi_metric_meta_30" ] || [ ${type} == "multi_metric_meta_ood_gru" ];then
#            for file in amazon_cds_train amazon_cds_ood_train amazon_cds_ood2_train
            for file in amazon_cds_train amazon_cds_ood_train  # 不要用ood2, ood2使用了target item
            do
              {
                sh "${file}.sh"
              }&
             done
          else
            sh amazon_cds_ood_train.sh
          fi
          cd ../../
        } &
      done
    } &
    done
    wait
#    sleep 960
    sleep 2100
}
done
wait # 等待所有任务结束
date
