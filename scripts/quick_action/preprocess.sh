index=$1
date=$2

sshpass -p DKIResearcher2022~ ssh TableSense@4.193.133.95 << EOF
cd /data/home/TableSense/raji/v-qinyuxu/llama-2-table
git pull
/spot/home/anaconda3/envs/llama/bin/python process_dataset/subtask.py --subtask_index $index
exit
EOF

cd /mnt/c/Users/v-qinyuxu/Desktop/data/daily_archieve/2023_9_$date
sshpass -p DKIResearcher2022~ rsync -av TableSense@4.193.133.95:/spot/v-qinyuxu/llama_dataset/ llama_dataset$index/
rsync -av llama_dataset$index/ FAREAST.hadong@gcrsandbox488.redmond.corp.microsoft.com:~/llama-2-table/src/llama_recipes/datasets/tablesense_dataset/subtask_$index