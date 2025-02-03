## setup environment
```bash
git clone https://github.com/Yan2266336/FinBen.git --recursive
conda create -n finben python=3.12
conda activate finben

cd FinBen/finlm_eval/
pip install -e .
pip install -e .[vllm]

# Login to your Huggingface
export HF_TOKEN="your_hf_token"
echo $HF_TOKEN
```

## model evaluation
```bash
./run_eval.sh
```
### for GPT model
```bash
lm_eval --model openai-chat-completions\
        --model_args "model=gpt-4o" \
        --tasks GRQAGen \
        --output_path results \
        --use_cache ./cache \
        --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
        --log_samples \
        --apply_chat_template \
        --include_path ./tasks
```

### for small model
```bash
lm_eval --model hf \
        --model_args "pretrained=Qwen/Qwen2.5-0.5B" \
        --tasks regAbbreviation \
        --num_fewshot 0 \
        --device cuda:1 \
        --batch_size 8 \
        --output_path results \
        --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-reAbbr-0shot-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
        --log_samples \
        --apply_chat_template \
        --include_path ./tasks
```

### for large model
```bash
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
```
```bash
lm_eval --model vllm \
        --model_args "pretrained=google/gemma-2-27b-it,tensor_parallel_size=4,gpu_memory_utilization=0.8,max_model_len=1024" \
        --tasks GRQA \
        --batch_size auto \
        --output_path results \
        --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
        --log_samples \
        --apply_chat_template \
        --include_path ./tasks
        
lm_eval --model vllm \
        --model_args "pretrained=meta-llama/Llama-3.2-1B-Instruct,tensor_parallel_size=4,gpu_memory_utilization=0.8,max_length=8192" \
        --tasks GRFNS2023 \
        --batch_size auto \
        --output_path results \
        --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
        --log_samples \
        --apply_chat_template \
        --include_path ./tasks
```
***results will be saved to FinBen/results/, you could add it into .gitignore***

- **0-shot setting:** Use `num_fewshot=0` and `lm-eval-results-gr-0shot` as the results repository.
- **5-shot setting:** Use `num_fewshot=5` and `lm-eval-results-gr-5shot` as the results repository.
- **Base models:** Remove `apply_chat_template`.
- **Instruction models:** Use `apply_chat_template`.

## add new task
```bash
cd FinBen/tasks/your_project_folder/ # create yaml file for new task
# Modify reg_utils.py to include metrics functions 

- **lm-evaluation-harness/docs/task_guide.md** # Good Reference Tasks
```

