import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from model_merging_methods.merging_methods import MergingMethod
from utils.utils import set_random_seed, smart_tokenizer_and_embedding_resize

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", type=str, default="")
    parser.add_argument("--m1", type=str, default="")
    parser.add_argument("--m2", type=str, default="")
    parser.add_argument("--m3", type=str, default="")
    parser.add_argument("--base-model-path", type=str, default="meta-llama/Llama-2-13b-hf")
    parser.add_argument("--method", type=str, default="")
    parser.add_argument("--DARE-method", type=str, default="")
    parser.add_argument("--DARE-rate", type=float, default=0)
    args = parser.parse_args()

    print(args)

    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=False)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model_path)

    tokenizer1 = AutoTokenizer.from_pretrained(args.m1, use_fast=False)
    tokenizer2 = AutoTokenizer.from_pretrained(args.m2, use_fast=False)
    model1 = AutoModelForCausalLM.from_pretrained(args.m1)
    model2 = AutoModelForCausalLM.from_pretrained(args.m2)

    if args.m3 != "":
        model3 = AutoModelForCausalLM.from_pretrained(args.m3)
        tokenizer3 = AutoTokenizer.from_pretrained(args.m3, use_fast=False)

    models_to_merge = [model1, model2] if args.m3 == "" else [model1, model2, model3]
    tokenizers = [tokenizer1, tokenizer2] if args.m3 == "" else [tokenizer1, tokenizer2, tokenizer3]

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token="[PAD]"),
        model=base_model,
        tokenizer=base_tokenizer
    )

    
    for finetuned_model, finetuned_tokenizer in zip(models_to_merge, tokenizers):
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]"),
            model=finetuned_model,
            tokenizer=finetuned_tokenizer
        )


    print('models loaded', flush=True)

    set_random_seed(seed=0)
    merging_method = MergingMethod(merging_method_name=args.method)
    merged_model = base_model
    merged_model = merging_method.get_merged_model(merged_model=merged_model,
                                                models_to_merge=models_to_merge,
                                                exclude_param_names_regex=[],
                                                trainers=[None, None],
                                                scaling_coefficient=None,
                                                nums_fisher_examples=None, 
                                                fisher_scaling_coefficients=None,
                                                normalize_fisher_weight=None,
                                                minimal_fisher_weight=None,
                                                nums_regmean_examples=None,
                                                reduce_non_diagonal_ratio=None,
                                                param_value_mask_rate=None,
                                                weight_format="delta_weight",
                                                weight_mask_rates=[args.DARE_rate,args.DARE_rate] ,
                                                use_weight_rescale=True,
                                                mask_strategy="random",
                                                mask_apply_method=args.DARE_method,
                                                models_use_deepcopy=False)

    for name in ["/merged_model", "/tokenizer_1", "/tokenizer_2", "/tokenizer_3"]:
        if not os.path.exists(args.save_path + name):
            os.makedirs(args.save_path + name)
    
    merged_model.save_pretrained(args.save_path + "/merged_model")
    tokenizer1.save_pretrained(args.save_path + "/tokenizer_1")
    tokenizer2.save_pretrained(args.save_path + "/tokenizer_2")

    if args.m3 != "":
        tokenizer3.save_pretrained(args.save_path + "/tokenizer_3")
    
if __name__ == "__main__":
    main()