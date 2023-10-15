date
#sleep 15000
#for dataset in cds electronic beauty
for dataset in cds
do
{
  for model in new_sasrec
    do
    {
#      for type in time_cost
      for type in time_cost_pred
      do
        {
          folder="${model}/${type}"
          echo ${folder}
          cd ${folder}
          if [ ${type} = "time_cost_pred" ];then
            for file in amazon_${dataset}_ood_lof_train amazon_${dataset}_ood_ocsvm_train amazon_${dataset}_ood_train  # 不要用ood2, ood2使用了target item
            do
              {
                sh "${file}.sh"
              }&
             done
#          else
#            sh amazon_${dataset}_ood_train.sh
##            echo amazon_${dataset}_ood_train.sh
          fi
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
