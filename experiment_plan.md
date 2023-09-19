To be checked:
* if special tokens are correctly tokenized

Experiment Plan:
* Only require the model to answer questions about number of header rows
* Use full-parameter training





Quick Command:
1. Copy Dataset from team server to local:  scp -r TableSense@4.193.133.95:/spot/v-qinyuxu/llama_dataset .
2. Copy Dataset from local to GCR:          rsync -av llama_dataset/ FAREAST.hadong@gcrsandbox488.redmond.corp.microsoft.com:~/*s/ft*/tablesense_dataset/
3. Copy trained model from GCR to local:    scp -r llama_dataset/ FAREAST.hadong@gcrsandbox488.redmond.corp.microsoft.com:~/*s/*save .
4. Start training:                          torchrun --nnodes 1 --nproc_per_node 4 llama_finetuning.py --enable_fsdp  --use_peft --peft_method lora --pure_bf16 --num_epochs 3
5. Inference:                               cat inference/tablesense_prompt.txt | python inference/inference.py --model_name Llama-2-7b-hf [--peft_model Llama-2-7b-save]
6. Large Scale Inference: