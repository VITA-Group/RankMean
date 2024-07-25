import jsonlines
import sys
import shutil
import logging
import os
from tqdm import tqdm
import glob
import json
import torch
import datasets
from vllm import LLM, SamplingParams
from human_eval.data import write_jsonl, read_problems, stream_jsonl

from utils.evaluate_llms_utils import batch_data, extract_answer_number, remove_boxed, last_boxed_only_string, process_results, \
    generate_instruction_following_task_prompt, get_math_task_prompt, generate_code_task_prompt, read_mbpp
from utils.load_config import cache_dir


def test_alpaca_eval(llm, finetuned_model_name, args, logger: logging.Logger, start_index=0, end_index=sys.maxsize,
                     save_model_path=None, save_gen_results_folder=None):
    try:
        eval_set = datasets.load_dataset(path=os.path.join(cache_dir, "alpaca_eval"), name="alpaca_eval")["eval"]
    except:
        eval_set = datasets.load_dataset(path="tatsu-lab/alpaca_eval", name="alpaca_eval", cache_dir=cache_dir)["eval"]

    instructions = []
    reference_outputs = []
    for example in eval_set:
        # dictionary with 'instruction', 'output': 'generator' and 'dataset' as keys
        instructions.append(example["instruction"])
        reference_outputs.append(example)

    instructions = instructions[start_index:end_index]
    reference_outputs = reference_outputs[start_index:end_index]

    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=2048)
    logger.info(f"sampling params is {sampling_params}")

    shutil.rmtree(save_gen_results_folder, ignore_errors=True)
    os.makedirs(save_gen_results_folder, exist_ok=True)
    generator_name = save_model_path if save_model_path is not None else finetuned_model_name
    logger.info(f"generator name is {generator_name}")

    for idx, (prompt, reference_output) in enumerate(zip(instructions, reference_outputs)):
        output_file = f"{save_gen_results_folder}/{start_index + idx}.jsonl"

        generated_outputs = []
        prompt = [generate_instruction_following_task_prompt(instruction=prompt, is_chat_model=True)]
        completions = llm.generate(prompt, sampling_params)
        for output in completions:
            generated_text = output.outputs[0].text
            generated_outputs.append({
                "instruction": reference_output["instruction"],
                "output": generated_text,
                "generator": generator_name,
                "dataset": reference_output["dataset"]
            })

        write_jsonl(output_file, generated_outputs)

    files = sorted(glob.glob(f"{save_gen_results_folder}/*.jsonl"))
    logger.info(f"find {len(files)} files in {save_gen_results_folder}")

    outputs = []
    for instruction_file in tqdm(files, total=len(files)):
        codes = [c for c in stream_jsonl(instruction_file)]
        outputs += codes

    logger.info(f"save to {save_gen_results_folder}.json")
    with open(f"{save_gen_results_folder}.json", "w", encoding="utf-8") as fout:
        json.dump(outputs, fout)
    if save_model_path is not None:
        shutil.rmtree(save_model_path, ignore_errors=True)

    del llm
    torch.cuda.empty_cache()


def test_hendrycks_math(llm, test_data_path, args, logger: logging.Logger, start_index=0, end_index=sys.maxsize, save_model_path=None):
    hendrycks_math_ins = []
    hendrycks_math_answers = []
    problem_prompt = get_math_task_prompt()
    logger.info(f"MATH test prompt is {problem_prompt}")
    with open(test_data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            temp_instr = problem_prompt.format(instruction=item["instruction"])
            hendrycks_math_ins.append(temp_instr)
            solution = item['output']
            temp_ans = remove_boxed(last_boxed_only_string(solution))
            hendrycks_math_answers.append(temp_ans)

    hendrycks_math_ins = hendrycks_math_ins[start_index:end_index]
    hendrycks_math_answers = hendrycks_math_answers[start_index:end_index]
    batch_hendrycks_math_ins = batch_data(hendrycks_math_ins, batch_size=50)

    stop_tokens = ["Instruction:", "Instruction", "Response:", "Response"]
    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=2048, stop=stop_tokens)
    logger.info(f"sampling params is {sampling_params}")

    res_completions = []
    for idx, prompt in enumerate(batch_hendrycks_math_ins):
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]
        completions = llm.generate(prompt, sampling_params)
        for output in completions:
            generated_text = output.outputs[0].text
            res_completions.append(generated_text)

    results = []
    invalid_outputs = []
    for idx, (prompt, completion, prompt_answer) in enumerate(zip(hendrycks_math_ins, res_completions, hendrycks_math_answers)):
        res = process_results(prompt, completion, prompt_answer, invalid_outputs)
        results.append(res)
    accuracy = sum(results) / len(results)
    logger.info(f"invalid outputs length is {len(invalid_outputs)}, invalid_outputs are {invalid_outputs}")
    logger.info(f"data index starts from {start_index}, ends at {end_index}")
    logger.info(f"MATH test data length is {len(results)}, accuracy is {accuracy}")
    logger.info(args)
    if save_model_path is not None:
        shutil.rmtree(save_model_path, ignore_errors=True)

    del llm
    torch.cuda.empty_cache()


def test_human_eval(llm, args, logger: logging.Logger, start_index=0, end_index=sys.maxsize, save_model_path=None, save_gen_results_folder=None):
    problems = read_problems()
    task_ids = sorted(problems.keys())[start_index: end_index]
    prompts = [problems[task_id]['prompt'] for task_id in task_ids]
    num_samples = len(prompts)
    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=2048)

    shutil.rmtree(save_gen_results_folder, ignore_errors=True)
    os.makedirs(save_gen_results_folder, exist_ok=True)

    for i in tqdm(range(num_samples), ncols=0, total=num_samples):
        output_file = f"{save_gen_results_folder}/{args.start_index + i}.jsonl"

        prompt = prompts[i].replace('    ', '\t')
        prompt_batch = [generate_code_task_prompt(prompt)]

        ids_batch = [task_ids[i]]
        completion_seqs = []

        loops = 1

        for _ in tqdm(range(loops), total=loops, leave=False, ncols=0):

            with torch.no_grad():
                completions = llm.generate(prompt_batch, sampling_params)
            gen_seqs = [completions[0].outputs[0].text]

            if gen_seqs is not None:
                assert len(ids_batch) == 1
                task_id = ids_batch[0]

                for seq_idx, gen_seq in enumerate(gen_seqs):
                    completion_seq = gen_seq.split("### Response:")[-1]
                    completion_seq = completion_seq.replace('\t', '    ')
                    all_code = gen_seq.replace('\t', '    ')

                    completion_seqs.append(
                        {'task_id': task_id,
                         'completion': completion_seq,
                         'all_code': all_code,
                         }
                    )

        write_jsonl(output_file, completion_seqs)

    files = sorted(glob.glob(f"{save_gen_results_folder}/*.jsonl"))
    logger.info(f"find {len(files)} files in {save_gen_results_folder}")

    outputs = []
    for code_file in tqdm(files, total=len(files)):
        codes = [c for c in stream_jsonl(code_file)]
        for code in codes:
            completion = code['completion']
            completion = completion.replace("\r", "")
            completion = completion.strip()
            if '```python' in completion:
                logger.info("completion matches ```python")
                def_line = completion.index('```python')
                completion = completion[def_line:].strip()
                completion = completion.replace('```python', '')
                try:
                    next_line = completion.index('```')
                    completion = completion[:next_line].strip()
                except:
                    logger.info("wrong completion")
            if "__name__ == \"__main__\"" in completion:
                logger.info("completion matches __name__ == \"__main__\"")
                try:
                    next_line = completion.index('if __name__ == "__main__":')
                    completion = completion[:next_line].strip()
                except:
                    logger.info("wrong completion")
            if "# Example usage" in completion:
                logger.info("completion matches # Example usage")
                next_line = completion.index('# Example usage')
                completion = completion[:next_line].strip()
            # the following codes are used to deal with the outputs of code-alpaca
            if "The solution is:" in completion:
                logger.info("completion matches The solution is:")
                def_line = completion.index("The solution is:")
                completion = completion[def_line:].strip()
                completion = completion.replace('The solution is:', '')
                try:
                    next_line = completion.index('\n\nThe answer is:')
                    completion = completion[:next_line].strip()
                except:
                    completion = completion.strip()
                    logger.info("maybe wrong completion")
            if "The answer is:" in completion:
                logger.info("completion matches The answer is:")
                def_line = completion.index("The answer is:")
                completion = completion[def_line:].strip()
                completion = completion.replace('The answer is:', '')
                try:
                    next_line = completion.index('\n\nThe answer is:')
                    completion = completion[:next_line].strip()
                except:
                    completion = completion.strip()
                    logger.info("maybe wrong completion")
            code['completion'] = completion
        outputs += codes

    logger.info(f"save to {save_gen_results_folder}.jsonl")
    write_jsonl(f"{save_gen_results_folder}.jsonl", outputs)
    if save_model_path is not None:
        shutil.rmtree(save_model_path, ignore_errors=True)

    del llm
    torch.cuda.empty_cache()


