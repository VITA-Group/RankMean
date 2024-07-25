import argparse
import sys
import logging
from vllm import LLM, SamplingParams

from evals import test_alpaca_eval, test_hendrycks_math, test_human_eval



if __name__ == "__main__":

    parser = argparse.ArgumentParser("Interface for direct inference merged LLMs")
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--save-path", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument("--end_index", type=int, default=sys.maxsize)
    parser.add_argument("--finetuned-model-name", type=str)
    
    args = parser.parse_args()

    llm = LLM(model=args.model_path, tensor_parallel_size=1)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if args.task == 'human_eval':   
        save_gen_results_folder = args.save_path
        test_human_eval(llm=llm, args=args, logger=logger,
                        save_model_path=None, save_gen_results_folder=save_gen_results_folder)
        
    elif args.task == "alpaca_eval":
        test_alpaca_eval(llm=llm, finetuned_model_name=args.finetuned_model_name,
                         args=args, logger=logger, start_index=args.start_index, end_index=args.end_index,
                         save_model_path=None, save_gen_results_folder=args.save_path)


    elif args.task == 'MATH':
        test_data_path = "math_code_data/MATH_test.jsonl"
        test_hendrycks_math(llm=llm, test_data_path=test_data_path, args=args, logger=logger,
                            start_index=args.start_index, end_index=args.end_index, save_model_path=None)
        

    sys.exit()