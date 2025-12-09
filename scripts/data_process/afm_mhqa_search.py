# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the AFM-MHQA-Agent-SFT-Dataset to Search-R1 format but save as JSON with instruction/input/output keys
"""

import re
import os
import json
import argparse

def make_prefix(question, template_type='base'):
    # Remove "Question: " prefix if it exists to avoid duplication with the template
    if question.lower().startswith("question:"):
        question = question[9:].strip()

    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
    else:
        raise NotImplementedError
    return prefix

def process_output(text):
    if not text:
        return ""
        
    # 1. Unify tags to <think>, <search>, <information>
    # <plan> -> <think>
    text = text.replace('<plan>', '<think>').replace('</plan>', '</think>')
    # <reflection> -> <think>
    text = text.replace('<reflection>', '<think>').replace('</reflection>', '</think>')
    # <wiki_search> -> <search>
    text = text.replace('<wiki_search>', '<search>').replace('</wiki_search>', '</search>')
    # <observation> -> <information>
    text = text.replace('<observation>', '<information>').replace('</observation>', '</information>')
    
    # 2. Merge adjacent <think> tags
    # Pattern: </think> followed by optional whitespace followed by <think>
    # We replace it with a newline to keep text separated but merged into one block
    pattern = r'</think>\s*<think>'
    text = re.sub(pattern, '\n', text)
    
    return text

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Default paths
    parser.add_argument('--input_file', default='../sft-datasets/AFM-MHQA-Agent-SFT-Dataset.json')
    parser.add_argument('--output_file', default='../sft-datasets/AFM-MHQA-Agent-SFT-Dataset-Processed.json')
    parser.add_argument('--template_type', type=str, default='base')
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.input_file}...")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    
    print(f"Loaded {len(data_list)} samples.")
    
    processed_data = []
    
    for item in data_list:
        # Original fields
        original_input = item.get('input', '')
        original_output = item.get('output', '')
        
        # 1. Format Prompt (instruction)
        # The user requested to convert instruction to the Search-R1 prompt format
        # and keep 'input' as is (the question).
        # However, the make_prefix function actually incorporates the question into the prompt.
        # The user said: "instruction转换成@Search-R1/ 的prompt格式，input也就是问题不用修改"
        # This implies:
        # instruction = Search-R1 template (make_prefix output, possibly WITHOUT the question appended if we want to keep structure strictly, OR WITH the question if that's the intended prompt)
        # Let's look at make_prefix: it takes 'question' and returns the FULL prompt including "Question: {question}".
        
        # If we want to match the typical SFT format where instruction + input = full prompt:
        # Option A: instruction = template text, input = question. 
        # Option B: instruction = full prompt (template + question), input = "" or duplicate.
        
        # Based on "input也就是问题不用修改" (input i.e. question, do not modify),
        # and "instruction转换成...prompt格式" (instruction convert to prompt format),
        # I will construct the 'instruction' field to contain the PROMPT TEMPLATE text, 
        # but wait, the Search-R1 make_prefix embeds the question at the end.
        
        # Let's adjust make_prefix to separate template and question if possible, OR:
        # The user likely wants the 'instruction' field to be the full system prompt/instruction that guides the model, 
        # and 'input' to be the specific instance question.
        
        # Search-R1 'make_prefix' returns: "Answer... Question: {question}\n"
        
        # Let's extract just the template part for 'instruction', or just put the full formatted thing in instruction?
        # "instruction转换成@Search-R1/ 的prompt格式" -> likely means the instruction field should become the Search-R1 system prompt.
        # The Search-R1 system prompt is:
        # "Answer the given question. You must conduct reasoning inside <think> ... Question: "
        # But usually 'input' is concatenated to 'instruction'.
        
        # Let's assume the goal is:
        # instruction: The Search-R1 description of tools and formatting.
        # input: The original question (unchanged).
        # output: The processed response.
        
        # Let's verify make_prefix behavior again. It returns the whole string.
        # I will construct a generic instruction string that matches the template minus the specific question.
        
        search_r1_template = """Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>."""

        # The user said "instruction转换成...prompt格式", and "input...不用修改".
        # So:
        new_instruction = search_r1_template
        new_input = original_input # Keep original input
        
        # However, the Search-R1 logic (make_prefix) appends "Question: {question}" at the end of the prompt.
        # If we just set instruction = template, and input = question, usually SFT frameworks join them like:
        # instruction + "\n" + input
        # Or instruction + input.
        
        # The Search-R1 make_prefix output ends with "Question: {question}\n".
        # If 'input' (original_input) is "Question: ...", then simply concatenating template + input roughly works.
        # But let's be precise.
        
        # If I strictly follow "instruction becomes Search-R1 prompt format":
        # I will use the template text.
        
        # 2. Process Output
        new_output = process_output(original_output)
        
        processed_item = {
            "instruction": new_instruction,
            "input": new_input,
            "output": new_output
        }
        processed_data.append(processed_item)
    
    # Save to JSON
    print(f"Saving processed data to {args.output_file}...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
            
    print("Done.")
