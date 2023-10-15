date
sleep 1800
#for((i=1;i<=5;i++))
#do
#{
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
          for file in movielens_train amazon_electronic_train amazon_cds_train amazon_beauty_train
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
#    sleep 1500
#}
#done
#wait # 等待所有任务结束
#date
