#!/bin/bash

set -e

# Run the Hugging Face LLM evaluation command
lm_eval --model hf \
        --model_args "pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
        --tasks Leaderboard_TSA \
        --num_fewshot 0 \
        --device cuda:1 \
        --batch_size 8 \
        --output_path results/Leaderboard \
        --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-llama-regulationXBRLTag-5shot-result,push_results_to_hub=False,push_samples_to_hub=False,public_repo=False" \
        --log_samples \
        --include_path ./tasks
        #--apply_chat_template \
        

# Run the Hugging Face VLLM evaluation command
# lm_eval --model vllm \
#         --model_args "pretrained=TheFinAI/FinLLaMA,tensor_parallel_size=4,gpu_memory_utilization=0.8,max_model_len=1024" \
#         --tasks regAbbreviation \
#         --num_fewshot 0 \
#         --batch_size auto \
#         --output_path results/abbr \
#         --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-llama-regulationAbbr-0shot-result,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
#         --log_samples \
#         --include_path ./tasks
#         #--apply_chat_template \

# output message
echo "Evaluation completed successfully!"