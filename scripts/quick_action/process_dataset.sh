sshpass -p DKIResearcher2022~ ssh TableSense@4.193.133.95 << EOF
cd /data/home/TableSense/raji/v-qinyuxu/llama-2-table
git pull
/spot/home/anaconda3/envs/llama/bin/python process_dataset/subtask1.py
exit
EOF
cd /mnt/c/Users/v-qinyuxu/Desktop/data/daily_archieve/2023_9_20
sshpass -p DKIResearcher2022~ scp -r TableSense@4.193.133.95:/spot/v-qinyuxu/llama_dataset .
rsync -av llama_dataset/ FAREAST.hadong@gcrsandbox488.redmond.corp.microsoft.com:~/llama-2-table/src/llama_recipes/datasets/tablesense_dataset/
cd /mnt/c/Users/v-qinyuxu/Desktop/llama-2-table