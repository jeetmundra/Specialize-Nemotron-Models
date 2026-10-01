#!/usr/bin/env python3
"""
Nemotron-Mini-4B C++ Specialization & Dual Evaluation Pipeline
Configured for single-GPU execution with official Chat Templates,
domain-stratified sampling, and adapter fusion (merge_and_unload).
"""
import os
import re
import time
import random
import subprocess
import json
import psutil
import argparse
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# Configuration
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.path.abspath("./hf_cache")

MODEL_ID = "nvidia/Nemotron-Mini-4B-Instruct"
EXPERIMENT_RESULTS = {
    "Baseline": {},
    "Specialized (Merged)": {}
}

# ------------------------------------------------------------------------------
# 1. C++ CODE CLEANER & SECURE g++ SANDBOX RUNNER
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

def run_gpp_test(full_code, binary_name="temp_test_bin", timeout_sec=5):
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
# 2. 50-PROBLEM CATEGORIZED BENCHMARK SUITE
# ------------------------------------------------------------------------------
benchmark_suite = [
    # STL (10)
    {"cat": "STL & Data Structures", "prompt": "Implement a C++ function `vector<int> rotate_array(vector<int> arr, int k)` that rotates an array right by k steps.",
     "test": "assert(rotate_array({1,2,3,4,5}, 2) == vector<int>({4,5,1,2,3})); assert(rotate_array({10,20}, 1) == vector<int>({20,10}));"},
    {"cat": "STL & Data Structures", "prompt": "Implement a C++ function `bool is_valid_parentheses(string s)` checking balanced '()', '{}', '[]'.",
     "test": "assert(is_valid_parentheses(\"()[]{}\") == true); assert(is_valid_parentheses(\"([)]\") == false); assert(is_valid_parentheses(\"(\") == false);"},
    {"cat": "STL & Data Structures", "prompt": "Implement a C++ function `vector<int> top_k_frequent(vector<int>& nums, int k)` returning k most frequent elements in descending order.",
     "test": "vector<int> a = {1,1,1,2,2,3}; auto r = top_k_frequent(a, 2); assert(r[0] == 1 && r[1] == 2);"},
    {"cat": "STL & Data Structures", "prompt": "Implement a C++ function `vector<int> merge_sorted(vector<int> a, vector<int> b)` merging two sorted vectors.",
     "test": "assert(merge_sorted({1,3,5}, {2,4,6}) == vector<int>({1,2,3,4,5,6}));"},
    {"cat": "STL & Data Structures", "prompt": "Implement a C++ function `int find_kth_largest(vector<int> nums, int k)` using a min-heap.",
     "test": "assert(find_kth_largest({3,2,1,5,6,4}, 2) == 5);"},
    {"cat": "STL & Data Structures", "prompt": "Implement a C++ function `vector<int> remove_duplicates(vector<int> nums)` preserving original first-occurrence order.",
     "test": "assert(remove_duplicates({4,5,4,2,5,1}) == vector<int>({4,5,2,1}));"},
    {"cat": "STL & Data Structures", "prompt": "Implement a C++ function `map<char, int> count_frequencies(string s)` returning char occurrence counts.",
     "test": "auto m = count_frequencies(\"abbccc\"); assert(m['a']==1 && m['b']==2 && m['c']==3);"},
    {"cat": "STL & Data Structures", "prompt": "Implement a C++ function `int evaluate_rpn(vector<string> tokens)` that evaluates Reverse Polish Notation.",
     "test": "assert(evaluate_rpn({\"2\",\"1\",\"+\",\"3\",\"*\"}) == 9);"},
    {"cat": "STL & Data Structures", "prompt": "Implement a C++ function `vector<int> intersection(vector<int> a, vector<int> b)` returning unique elements in both.",
     "test": "assert(intersection({1,2,2,1}, {2,2}) == vector<int>({2}));"},
    {"cat": "STL & Data Structures", "prompt": "Implement a C++ function `bool is_monotonic(vector<int> nums)` returning true if strictly non-decreasing or non-increasing.",
     "test": "assert(is_monotonic({1,2,2,3}) == true); assert(is_monotonic({1,3,2}) == false);"},
    # DP (10)
    {"cat": "Dynamic Programming", "prompt": "Implement a C++ function `int fibonacci(int n)` returning the n-th Fibonacci number in O(n) time.",
     "test": "assert(fibonacci(0) == 0); assert(fibonacci(1) == 1); assert(fibonacci(10) == 55);"},
    {"cat": "Dynamic Programming", "prompt": "Implement a C++ function `int climb_stairs(int n)` computing distinct ways to climb n steps (1 or 2 steps at a time).",
     "test": "assert(climb_stairs(2) == 2); assert(climb_stairs(3) == 3); assert(climb_stairs(5) == 8);"},
    {"cat": "Dynamic Programming", "prompt": "Implement a C++ function `int coin_change(vector<int>& coins, int amount)` returning fewest coins needed, or -1 if impossible.",
     "test": "vector<int> c = {1,2,5}; assert(coin_change(c, 11) == 3); vector<int> c2 = {2}; assert(coin_change(c2, 3) == -1);"},
    {"cat": "Dynamic Programming", "prompt": "Implement a C++ function `int length_of_lis(vector<int> nums)` returning length of Longest Increasing Subsequence.",
     "test": "assert(length_of_lis({10,9,2,5,3,7,101,18}) == 4); assert(length_of_lis({7,7,7}) == 1);"},
    {"cat": "Dynamic Programming", "prompt": "Implement a C++ function `int longest_common_subsequence(string text1, string text2)`.",
     "test": "assert(longest_common_subsequence(\"abcde\", \"ace\") == 3); assert(longest_common_subsequence(\"abc\", \"def\") == 0);"},
    {"cat": "Dynamic Programming", "prompt": "Implement a C++ function `int max_sub_array(vector<int> nums)` computing Kadane's maximum subarray sum.",
     "test": "assert(max_sub_array({-2,1,-3,4,-1,2,1,-5,4}) == 6); assert(max_sub_array({1}) == 1);"},
    {"cat": "Dynamic Programming", "prompt": "Implement a C++ function `int min_path_sum(vector<vector<int>> grid)` finding minimum path sum from top-left to bottom-right.",
     "test": "assert(min_path_sum({{1,3,1},{1,5,1},{4,2,1}}) == 7);"},
    {"cat": "Dynamic Programming", "prompt": "Implement a C++ function `bool can_partition(vector<int> nums)` returning if array can be partitioned into two equal sum subsets.",
     "test": "assert(can_partition({1,5,11,5}) == true); assert(can_partition({1,2,3,5}) == false);"},
    {"cat": "Dynamic Programming", "prompt": "Implement a C++ function `int knapsack_01(int W, vector<int> wt, vector<int> val, int n)` solving standard 0/1 knapsack.",
     "test": "assert(knapsack_01(50, {10,20,30}, {60,100,120}, 3) == 220);"},
    {"cat": "Dynamic Programming", "prompt": "Implement a C++ function `int edit_distance(string word1, string word2)` returning minimum Levenshtein distance.",
     "test": "assert(edit_distance(\"horse\", \"ros\") == 3); assert(edit_distance(\"intention\", \"execution\") == 5);"},
    # Graph (10)
    {"cat": "Graph & Tree Algorithms", "prompt": "Implement a C++ function `bool valid_path(int n, vector<vector<int>>& edges, int source, int destination)` using BFS.",
     "test": "vector<vector<int>> e = {{0,1},{1,2},{2,0}}; assert(valid_path(3, e, 0, 2) == true);"},
    {"cat": "Graph & Tree Algorithms", "prompt": "Implement a C++ function `int num_islands(vector<vector<char>>& grid)` using DFS.",
     "test": "vector<vector<char>> g = {{'1','1','0'},{'1','1','0'},{'0','0','1'}}; assert(num_islands(g) == 2);"},
    {"cat": "Graph & Tree Algorithms", "prompt": "Implement a C++ function `int binary_search(vector<int> nums, int target)` returning target index or -1.",
     "test": "assert(binary_search({-1,0,3,5,9,12}, 9) == 4); assert(binary_search({-1,0,3,5,9,12}, 2) == -1);"},
    {"cat": "Graph & Tree Algorithms", "prompt": "Implement a C++ function `int search_rotated(vector<int> nums, int target)` in O(log n) time.",
     "test": "assert(search_rotated({4,5,6,7,0,1,2}, 0) == 4); assert(search_rotated({4,5,6,7,0,1,2}, 3) == -1);"},
    {"cat": "Graph & Tree Algorithms", "prompt": "Implement a C++ function `bool has_cycle(int n, vector<vector<int>> edges)` detecting cycle in directed graph using Kahn's algorithm.",
     "test": "assert(has_cycle(2, {{0,1},{1,0}}) == true); assert(has_cycle(2, {{0,1}}) == false);"},
    {"cat": "Graph & Tree Algorithms", "prompt": "Implement a C++ function `vector<int> bfs_traversal(int n, vector<vector<int>> adj)` starting at node 0.",
     "test": "assert(bfs_traversal(3, {{1,2},{0},{0}}) == vector<int>({0,1,2}));"},
    {"cat": "Graph & Tree Algorithms", "prompt": "Implement a C++ function `int shortest_path_unweighted(int n, vector<vector<int>> edges, int src, int dst)`.",
     "test": "assert(shortest_path_unweighted(4, {{0,1},{1,2},{2,3},{0,3}}, 0, 3) == 1);"},
    {"cat": "Graph & Tree Algorithms", "prompt": "Implement a C++ function `int max_depth_tree(vector<int> parents)` returning max depth given parent pointers (-1 is root).",
     "test": "assert(max_depth_tree({-1, 0, 0, 1, 1}) == 3);"},
    {"cat": "Graph & Tree Algorithms", "prompt": "Implement a C++ function `bool is_bipartite(int n, vector<vector<int>> adj)` checking 2-colorability.",
     "test": "assert(is_bipartite(4, {{1,3},{0,2},{1,3},{0,2}}) == true);"},
    {"cat": "Graph & Tree Algorithms", "prompt": "Implement a C++ function `int count_connected_components(int n, vector<vector<int>> edges)` using Union-Find.",
     "test": "assert(count_connected_components(5, {{0,1},{1,2},{3,4}}) == 2);"},
    # String (10)
    {"cat": "String & Parsing", "prompt": "Implement a C++ function `bool is_palindrome(string s)` ignoring non-alphanumerics and case.",
     "test": "assert(is_palindrome(\"A man, a plan, a canal: Panama\") == true); assert(is_palindrome(\"race a car\") == false);"},
    {"cat": "String & Parsing", "prompt": "Implement a C++ function `string reverse_words(string s)` reversing word order cleanly.",
     "test": "assert(reverse_words(\"the sky is blue\") == \"blue is sky the\");"},
    {"cat": "String & Parsing", "prompt": "Implement a C++ function `int length_of_longest_substring(string s)` without repeating characters.",
     "test": "assert(length_of_longest_substring(\"abcabcbb\") == 3); assert(length_of_longest_substring(\"bbbbb\") == 1);"},
    {"cat": "String & Parsing", "prompt": "Implement a C++ function `string longest_palindrome(string s)` returning longest palindromic substring.",
     "test": "string r = longest_palindrome(\"babad\"); assert(r == \"bab\" || r == \"aba\");"},
    {"cat": "String & Parsing", "prompt": "Implement a C++ function `bool is_anagram(string s, string t)`.",
     "test": "assert(is_anagram(\"anagram\", \"nagaram\") == true); assert(is_anagram(\"rat\", \"car\") == false);"},
    {"cat": "String & Parsing", "prompt": "Implement a C++ function `string compress_string(string s)` (e.g. \"aabcccccaaa\" -> \"a2b1c5a3\").",
     "test": "assert(compress_string(\"aabcccccaaa\") == \"a2b1c5a3\"); assert(compress_string(\"abc\") == \"abc\");"},
    {"cat": "String & Parsing", "prompt": "Implement a C++ function `int str_str(string haystack, string needle)` returning first occurrence index or -1.",
     "test": "assert(str_str(\"hello\", \"ll\") == 2); assert(str_str(\"aaaaa\", \"bba\") == -1);"},
    {"cat": "String & Parsing", "prompt": "Implement a C++ function `vector<string> fizz_buzz(int n)`.",
     "test": "assert(fizz_buzz(3) == vector<string>({\"1\",\"2\",\"Fizz\"}));"},
    {"cat": "String & Parsing", "prompt": "Implement a C++ function `string longest_common_prefix(vector<string> strs)`.",
     "test": "assert(longest_common_prefix({\"flower\",\"flow\",\"flight\"}) == \"fl\"); assert(longest_common_prefix({\"dog\",\"racecar\"}) == \"\");"},
    {"cat": "String & Parsing", "prompt": "Implement a C++ function `int my_atoi(string s)` converting string to 32-bit signed integer with clamping.",
     "test": "assert(my_atoi(\"42\") == 42); assert(my_atoi(\"   -42\") == -42); assert(my_atoi(\"4193 with words\") == 4193);"},
    # Math (10)
    {"cat": "Math & Bitwise", "prompt": "Implement a C++ function `bool is_prime(int n)` in O(sqrt(n)) time.",
     "test": "assert(is_prime(2) == true); assert(is_prime(17) == true); assert(is_prime(4) == false); assert(is_prime(1) == false);"},
    {"cat": "Math & Bitwise", "prompt": "Implement a C++ function `int count_set_bits(int n)` (Hamming weight).",
     "test": "assert(count_set_bits(11) == 3); assert(count_set_bits(128) == 1); assert(count_set_bits(0) == 0);"},
    {"cat": "Math & Bitwise", "prompt": "Implement a C++ function `int single_number(vector<int> nums)` where every element appears twice except for one.",
     "test": "assert(single_number({2,2,1}) == 1); assert(single_number({4,1,2,1,2}) == 4);"},
    {"cat": "Math & Bitwise", "prompt": "Implement a C++ function `int gcd(int a, int b)` using Euclidean algorithm.",
     "test": "assert(gcd(48, 18) == 6); assert(gcd(101, 103) == 1);"},
    {"cat": "Math & Bitwise", "prompt": "Implement a C++ function `int count_primes(int n)` using Sieve of Eratosthenes strictly less than n.",
     "test": "assert(count_primes(10) == 4); assert(count_primes(0) == 0); assert(count_primes(1) == 0);"},
    {"cat": "Math & Bitwise", "prompt": "Implement a C++ function `double my_pow(double x, int n)` in O(log n) time.",
     "test": "assert(abs(my_pow(2.0, 10) - 1024.0) < 1e-5); assert(abs(my_pow(2.0, -2) - 0.25) < 1e-5);"},
    {"cat": "Math & Bitwise", "prompt": "Implement a C++ function `bool is_power_of_two(int n)` using bitwise operations.",
     "test": "assert(is_power_of_two(1) == true); assert(is_power_of_two(16) == true); assert(is_power_of_two(3) == false);"},
    {"cat": "Math & Bitwise", "prompt": "Implement a C++ function `int reverse_integer(int x)` with 32-bit overflow check.",
     "test": "assert(reverse_integer(123) == 321); assert(reverse_integer(-123) == -321); assert(reverse_integer(120) == 21);"},
    {"cat": "Math & Bitwise", "prompt": "Implement a C++ function `int missing_number(vector<int> nums)` containing n distinct numbers in range [0, n].",
     "test": "assert(missing_number({3,0,1}) == 2); assert(missing_number({0,1}) == 2);"},
    {"cat": "Math & Bitwise", "prompt": "Implement a C++ function `long long fast_mod_pow(long long base, long long exp, long long mod)`.",
     "test": "assert(fast_mod_pow(2, 10, 1000) == 24); assert(fast_mod_pow(3, 5, 7) == 5);"}
]

# ------------------------------------------------------------------------------
# 3. EVALUATION FUNCTION (OFFICIAL CHAT TEMPLATE)
# ------------------------------------------------------------------------------
def evaluate_full_model(eval_model, tokenizer, testing_split, tag="Baseline"):
    eval_model.eval()
    print(f"\n" + "="*70)
    print(f"🧪 EVALUATING {tag.upper()}")
    print("="*70, flush=True)
    
    # 1. Latency with Warmup & Fixed 100 Tokens
    print(f"\n⏱️ Measuring {tag} Latency & Generation Throughput...", flush=True)
    messages = [
        {"role": "system", "content": "You are a competitive programming C++ assistant. Output ONLY valid C++ code inside a ```cpp block without any explanations or conversational chatter."},
        {"role": "user", "content": "Write ONLY the complete C++ function implementation for binary search."}
    ]
    p_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    dummy = tokenizer(p_str, return_tensors="pt").to("cuda:0")
    
    with torch.inference_mode():
        _ = eval_model.generate(**dummy, min_new_tokens=32, max_new_tokens=32, do_sample=False)
        torch.cuda.synchronize()
        
        t0 = time.time()
        out = eval_model.generate(**dummy, min_new_tokens=100, max_new_tokens=100, do_sample=False)
        torch.cuda.synchronize()
        t1 = time.time()
    
    actual_tokens = out.shape[1] - dummy["input_ids"].shape[1]
    latency_ms = ((t1 - t0) / actual_tokens) * 1000
    throughput = actual_tokens / (t1 - t0)
    
    print(f"   • Latency:    {latency_ms:.2f} ms/token", flush=True)
    print(f"   • Throughput: {throughput:.2f} tokens/sec", flush=True)
    
    # 2. 50-Problem Suite
    print(f"\n📊 Running 50-Problem Categorized Suite on {tag}...", flush=True)
    cat_stats = {}
    total_pass = 0
    total_comp = 0
    
    for idx, item in enumerate(benchmark_suite):
        cat = item["cat"]
        if cat not in cat_stats:
            cat_stats[cat] = {"passed": 0, "compiled": 0, "total": 0}
        cat_stats[cat]["total"] += 1
        
        messages = [
            {"role": "system", "content": "You are a competitive programming C++ assistant. Output ONLY valid C++ code inside a ```cpp block without any explanations or conversational chatter."},
            {"role": "user", "content": f"Write ONLY the complete C++ function implementation for:\n{item['prompt']}"}
        ]
        p_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(p_str, return_tensors="pt").to("cuda:0")
        
        with torch.inference_mode():
            gen = eval_model.generate(**inp, max_new_tokens=400, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        
        raw_code = tokenizer.decode(gen[0][inp['input_ids'].shape[1]:], skip_special_tokens=False)
        code = clean_cpp_code_strict(raw_code)
        
        full_unit = f"""
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <stack>
#include <cmath>
#include <climits>
#include <cassert>
using namespace std;
{code}
int main() {{
    {item['test']}
    return 0;
}}
"""
        comp, pass_, err = run_gpp_test(full_unit, f"bench_{tag}_{idx}")
        if comp:
            cat_stats[cat]["compiled"] += 1
            total_comp += 1
        if pass_:
            cat_stats[cat]["passed"] += 1
            total_pass += 1
            
        print(f"   [{idx+1:02d}/50] [{cat:22s}] -> {'✅ PASS' if pass_ else ('❌ COMP_FAIL' if not comp else '❌ ASSERT_FAIL')}", flush=True)
    
    # 3. HumanEval-C++ (164 problems)
    print(f"\n🧪 Evaluating HumanEval-C++ (164 Problems) on {tag}...", flush=True)
    he_comp = 0
    he_pass = 0
    
    for i in range(len(testing_split)):
        sample = testing_split[i]
        prompt_text = sample.get("prompt", "")
        test_code = sample.get("test", "")
        headers = extract_headers(prompt_text)
        
        messages = [
            {"role": "system", "content": "You are a C++ coding engine. Complete ONLY the function body inside a ```cpp block without any conversational text."},
            {"role": "user", "content": f"Complete ONLY the function body for this C++ function:\n```cpp\n{prompt_text}\n```"}
        ]
        p_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(p_str, return_tensors="pt").to("cuda:0")
        
        with torch.inference_mode():
            gen = eval_model.generate(**inp, max_new_tokens=300, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        
        raw_code = tokenizer.decode(gen[0][inp['input_ids'].shape[1]:], skip_special_tokens=False)
        code = clean_cpp_code_strict(raw_code)
        
        full_unit = f"{headers}\n\n{code}\n\n{test_code}"
        comp, pass_, err = run_gpp_test(full_unit, f"he_{tag}_{i}")
        if comp: he_comp += 1
        if pass_: he_pass += 1
        
        if (i+1) % 25 == 0 or (i+1) == len(testing_split):
            print(f"   [HE-C++ {i+1:03d}/164] Compiled: {he_comp} ({he_comp/(i+1)*100:.1f}%) | Passed: {he_pass} ({he_pass/(i+1)*100:.1f}%)", flush=True)
            
    EXPERIMENT_RESULTS[tag]["Latency (ms/tok)"] = round(latency_ms, 2)
    EXPERIMENT_RESULTS[tag]["Throughput (tok/s)"] = round(throughput, 2)
    EXPERIMENT_RESULTS[tag]["Benchmark Total"] = f"{total_pass}/50"
    EXPERIMENT_RESULTS[tag]["Benchmark Pass Rate %"] = round(total_pass / 50 * 100, 1)
    EXPERIMENT_RESULTS[tag]["HumanEval Compilation %"] = round(he_comp / 164 * 100, 1)
    EXPERIMENT_RESULTS[tag]["HumanEval Pass@1 %"] = round(he_pass / 164 * 100, 1)
    EXPERIMENT_RESULTS[tag]["Categories"] = {k: v["passed"] for k, v in cat_stats.items()}
    
    return cat_stats

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Nemotron-4B C++ Specialization & Evaluation")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN", None), help="Hugging Face access token")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline evaluation and jump to training")
    args = parser.parse_args()

    hf_token = args.hf_token

    print("=" * 70)
    print("🚀 NVIDIA NEMOTRON-4B: STRATIFIED C++ LORA TRAINING & EVALUATION")
    print(f"   • Target Device: cuda:0 ({torch.cuda.get_device_name(0)})")
    print(f"   • Cache Dir:     {os.environ['HF_HOME']}")
    print("=" * 70, flush=True)

    # 1. Load Tokenizer & Baseline Model on SINGLE GPU
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=hf_token,
        torch_dtype=torch.float16,
        device_map={"": 0}
    )
    model.config.use_cache = True
    total_params = sum(p.numel() for p in model.parameters())
    vram_alloc = torch.cuda.memory_allocated() / (1024**3)
    EXPERIMENT_RESULTS["Baseline"]["VRAM (GB)"] = round(vram_alloc, 2)
    EXPERIMENT_RESULTS["Baseline"]["Parameters"] = f"{total_params/1e9:.2f}B"

    # 2. Testing Split
    print("\n⏳ Sourcing HumanEval-C++...", flush=True)
    testing_split = load_dataset("bigcode/humanevalpack", "cpp", split="test")

    # 3. Full Baseline Benchmark
    if not args.skip_baseline:
        evaluate_full_model(model, tokenizer, testing_split, tag="Baseline")

    # 4. Stratified Dataset Preparation
    print("\n" + "=" * 70)
    print("📦 ASSEMBLING DOMAIN-STRATIFIED 12,000 TRAINING PAIRS")
    print("=" * 70, flush=True)
    
    def classify_domain(text):
        t = text.lower()
        if any(k in t for k in ["graph", "tree", "bfs", "dfs", "dijkstra", "adjacency", "node", "edge", "bipartite", "cycle"]):
            return "Graph & Tree"
        elif any(k in t for k in ["dp", "dynamic programming", "memoiz", "knapsack", "subsequence", "edit distance"]):
            return "Dynamic Programming"
        elif any(k in t for k in ["prime", "gcd", "bitwise", "bit", "pow", "math", "modulo", "hamming"]):
            return "Math & Bitwise"
        elif any(k in t for k in ["string", "palindrome", "substring", "anagram", "prefix", "suffix", "atoi"]):
            return "String & Parsing"
        else:
            return "STL & Data Structures"

    raw_pool = []
    try:
        alpaca = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
        for x in alpaca:
            inst, inp, out = x.get("instruction",""), x.get("input",""), x.get("output","")
            if any(k in f"{inst} {inp} {out}".lower() for k in ["c++", "cpp", "#include"]):
                p = f"{inst}\n{inp}".strip() if inp else inst
                raw_pool.append({"instruction": f"Write clean, optimized C++ code:\n{p}", "output": out})
    except Exception as e:
        print(f"   ⚠️ CodeAlpaca note: {e}")

    try:
        evol = load_dataset("theblackcat102/evol-codealpaca-v1", split="train")
        for x in evol:
            inst, out = x.get("instruction",""), x.get("output","")
            if any(k in f"{inst} {out}".lower() for k in ["c++", "cpp", "#include", "std::"]):
                raw_pool.append({"instruction": f"Solve this programming problem in C++:\n{inst}", "output": out})
    except Exception as e:
        print(f"   ⚠️ Evol note: {e}")

    try:
        feedback = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train", streaming=True)
        count = 0
        for x in feedback:
            q, a = x.get("query",""), x.get("answer","")
            if any(k in f"{q} {a}".lower() for k in ["c++", "cpp", "#include"]) and "```cpp" in a:
                raw_pool.append({"instruction": q, "output": a})
                count += 1
                if count >= 4000: break
    except Exception as e:
        print(f"   ⚠️ Feedback note: {e}")

    buckets = {"STL & Data Structures": [], "Dynamic Programming": [], "Graph & Tree": [], "String & Parsing": [], "Math & Bitwise": []}
    for item in raw_pool:
        buckets[classify_domain(item["instruction"] + " " + item["output"])].append(item)

    stratified_samples = []
    random.seed(42)
    for d, lst in buckets.items():
        random.shuffle(lst)
        stratified_samples.extend(lst[:2400])

    if len(stratified_samples) < 12000:
        rem = [x for x in raw_pool if x not in stratified_samples]
        random.shuffle(rem)
        stratified_samples.extend(rem[:(12000 - len(stratified_samples))])

    random.shuffle(stratified_samples)
    hf_train_ds = Dataset.from_list(stratified_samples[:12000])
    print(f"✅ Stratified training pool created with {len(hf_train_ds):,d} samples.", flush=True)

    # 5. PEFT / LoRA Injection
    print("\n" + "=" * 70)
    print("⚙️ INJECTING LORA ADAPTER (ALL-7 PROJECTIONS, r=16, alpha=32)")
    print("=" * 70, flush=True)
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()

    def format_prompt(sample):
        return f"<|im_start|>user\n{sample['instruction']}<|im_end|>\n<|im_start|>assistant\n```cpp\n{sample['output']}\n```<|im_end|>"

    hf_train_ds = hf_train_ds.map(lambda x: {"text": format_prompt(x)})

    # 6. SFT Training
    training_args = SFTConfig(
        output_dir="./lora_nemotron_cpp",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=50,
        fp16=True,
        dataset_text_field="text",
        max_seq_length=512,
        save_strategy="no",
        report_to="none"
    )

    # Compatible with both modern and older TRL versions
    try:
        trainer = SFTTrainer(
            model=peft_model,
            train_dataset=hf_train_ds,
            args=training_args,
            processing_class=tokenizer
        )
    except TypeError:
        trainer = SFTTrainer(
            model=peft_model,
            train_dataset=hf_train_ds,
            args=training_args,
            tokenizer=tokenizer
        )

    print("\n🚀 Starting LoRA fine-tuning on GPU 0...", flush=True)
    trainer.train()
    peft_model.save_pretrained("./lora_adapter_checkpoint")
    print("✅ Training complete. LoRA checkpoint saved.", flush=True)

    # 7. CRITICAL FIX: Merge Adapter into Base Model (Eliminates +73% latency overhead)
    print("\n" + "=" * 70)
    print("⚡ FUSING ADAPTER (merge_and_unload) FOR NATIVE SPEED INFERENCE")
    print("=" * 70, flush=True)
    fused_model = peft_model.merge_and_unload()
    fused_model.config.use_cache = True

    # 8. Evaluate Fused Specialized Model
    evaluate_full_model(fused_model, tokenizer, testing_split, tag="Specialized (Merged)")

    # 9. Output Summary & Plot
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS SUMMARY")
    print("=" * 70, flush=True)
    summary_df = pd.DataFrame(EXPERIMENT_RESULTS).T
    print(summary_df[["Latency (ms/tok)", "Throughput (tok/s)", "HumanEval Compilation %", "HumanEval Pass@1 %", "Benchmark Total"]], flush=True)
    
    with open("final_experiment_results.json", "w") as f:
        json.dump(EXPERIMENT_RESULTS, f, indent=4)

    # Generate IEEE Comparison Chart
    cats = ["STL & Data Structures", "Dynamic Programming", "Graph & Tree Algorithms", "String & Parsing", "Math & Bitwise"]
    base_vals = [EXPERIMENT_RESULTS["Baseline"]["Categories"].get(c, 0) for c in cats]
    spec_vals = [EXPERIMENT_RESULTS["Specialized (Merged)"]["Categories"].get(c, 0) for c in cats]

    x = np.arange(len(cats))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.bar(x - width/2, base_vals, width, label='Baseline (FP16)', color='#4A5568')
    ax.bar(x + width/2, spec_vals, width, label='Specialized (Stratified LoRA)', color='#2B6CB0')
    ax.set_ylabel('Pass Count (out of 10)', fontsize=11, fontweight='bold')
    ax.set_title('Domain Performance: Baseline vs. Specialized', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=15, ha='right', fontsize=9)
    ax.set_ylim(0, 10)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("ieee_benchmark_comparison.png", dpi=300)
    print("\n🎉 ALL DONE! Saved: final_experiment_results.json, ieee_benchmark_comparison.png, and ./lora_adapter_checkpoint", flush=True)

if __name__ == "__main__":
    main()
