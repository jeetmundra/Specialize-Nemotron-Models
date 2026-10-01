#!/usr/bin/env python3
"""
Standalone Benchmark Evaluation Script for Nemotron-Mini-4B C++
Supports evaluating both base models and LoRA-adapted/fused models
across the 50-problem categorized suite and 164-problem HumanEval-C++.
"""
import os
import re
import time
import subprocess
import json
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# ------------------------------------------------------------------------------
# C++ Extractions & Sandbox Test
# ------------------------------------------------------------------------------
def clean_cpp_code_strict(raw_text):
    raw_text = raw_text.replace("</s>", "").replace("<|im_end|>", "").replace("<|im_start|>", "")
    cpp_match = re.search(r'```(?:cpp|c\+\+|c)?\s*(.*?)(?:```|$)', raw_text, re.DOTALL)
    if cpp_match:
        return cpp_match.group(1).strip()
    return raw_text.strip()

def extract_headers(prompt_text):
    headers = []
    for line in prompt_text.split('\n'):
        s = line.strip()
        if s.startswith('#include') or s.startswith('using namespace'):
            headers.append(s)
    return '\n'.join(headers)

def run_gpp_test(full_code, binary_name="temp_eval_bin", timeout_sec=5):
    src_file = f"{binary_name}.cpp"
    with open(src_file, "w", encoding="utf-8") as f:
        f.write(full_code)
    
    comp = subprocess.run(
        ["g++", "-std=c++17", "-O2", src_file, "-o", f"./{binary_name}"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if comp.returncode != 0:
        return False, False, comp.stderr[:300]
    
    try:
        run = subprocess.run(
            [f"./{binary_name}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout_sec
        )
        return True, run.returncode == 0, ""
    except subprocess.TimeoutExpired:
        return True, False, "Execution Timeout"
    finally:
        for fpath in [src_file, f"./{binary_name}"]:
            if os.path.exists(fpath):
                try: os.remove(fpath)
                except Exception: pass

# ------------------------------------------------------------------------------
# Benchmark Suite
# ------------------------------------------------------------------------------
from train_nemotron_cpp import benchmark_suite, evaluate_full_model

def main():
    parser = argparse.ArgumentParser(description="Evaluate Nemotron-4B C++ Models")
    parser.add_argument("--base-model", type=str, default="nvidia/Nemotron-Mini-4B-Instruct", help="Base model HuggingFace ID")
    parser.add_argument("--adapter-path", type=str, default=None, help="Path to LoRA checkpoint to merge and evaluate")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN", None), help="HuggingFace token")
    parser.add_argument("--tag", type=str, default="Model_Eval", help="Tag for results")
    args = parser.parse_args()

    print(f"🚀 Loading base model: {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=args.hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        token=args.hf_token,
        torch_dtype=torch.float16,
        device_map={"": 0}
    )

    if args.adapter_path:
        print(f"⚡ Loading and merging LoRA adapter from {args.adapter_path}...")
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model = model.merge_and_unload()
        print("✅ Adapter successfully fused (merge_and_unload).")

    model.config.use_cache = True
    testing_split = load_dataset("bigcode/humanevalpack", "cpp", split="test")

    stats = evaluate_full_model(model, tokenizer, testing_split, tag=args.tag)
    print("\n✅ Evaluation finished successfully!")

if __name__ == "__main__":
    main()
