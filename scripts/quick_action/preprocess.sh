sshpass -p DKIResearcher2022~ ssh a << EOF
cd /data/home/TableSense/raji/v-qinyuxu/llama-2-table
git pull
/spot/home/anaconda3/envs/llama/bin/python process_dataset/subtask.py --model_name Llama-2-13b-hf
exit
EOF

sshpass -p DKIResearcher2022~ rsync -av TableSense@4.193.133.95:/spot/v-qinyuxu/llama_dataset/ ~/Desktop/msra/llama_finetuning/data/llama_dataset_all/
rsync -av ~/Desktop/msra/llama_finetuning/data/llama_dataset_all/llama_dataset  FAREAST.hadong@gcrsandbox488.redmond.corp.microsoft.com:~/llama-2-table/src/llama_recipes/datasets/tablesense_dataset/subtask_all
rsync -av ~/Desktop/msra/llama_finetuning/data/llama_dataset_all/llama_dataset/test_263_row_feature.json FAREAST.hadong@gcrsandbox488.redmond.corp.microsoft.com:~/llama-2-table/saved_result/result.json