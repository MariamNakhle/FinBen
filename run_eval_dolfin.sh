#!/bin/bash

set -e

# Run the Hugging Face LLM evaluation command
lm_eval --model hf \
        --model_args "pretrained=Qwen/Qwen2.5-7B-Instruct" \
        --tasks Dolfin \
        --device cuda:1 \
        --batch_size 1 \
        --output_path results/Leaderboard_dolfin_comet_slide \
        --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-llama-regulationXBRLTag-5shot-result,push_results_to_hub=False,push_samples_to_hub=False,public_repo=False" \
        --log_samples \
        --include_path ./tasks \
        --limit 20 \
        --apply_chat_template \


# output message
echo "Evaluation completed successfully!"
