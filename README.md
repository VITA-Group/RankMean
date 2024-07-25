## Introduction

This repository contain the code for the paper RankMean: Module-Level Importance Score for Merging Fine-tuned Large Language Models (ACL 2024).

If you find any problems, or have any questions, feel free to contact gabrieljp (at) usp (dot) br.
## Installation

To create a conda environment with the necessary dependencies, please run 
```
conda env create -f env.yaml
``` 
Afther that, please follow the [human_eval repository](https://github.com/openai/human-eval) to install the neecssary dependencies to run the human_eval evaluation. The same is necessary for the [alpaca_eval](https://github.com/tatsu-lab/alpaca_eval) evaluation.

Finally, it is necessary to change the ```cache_dir``` in ```utils/load_config.py```.

## Merging

To merge two models, please run 
```
python merge.py --m1 <model1_path> --m2 <model2_path> --base-model-path <base_model_path> --method <merging_method> --save-path <save_path>
``` 
where the ```<merging_method>``` used in our work were: ```rankmean```, ```finer_grained```, ```reversed```, ```average_merging``` and ```mask_merging``` (DARE).

If using DARE, it is also necessary to pass parameters ```--DARE-method``` (method to use after prunning) and ```--DARE-rate``` (prunning rate).

## Evaluation

To perform the evaluations of human_eval, alpaca_eval and MATH, please run 
```
python inference.py --model-path <model_path> --save-path <save_path> --task <task_to_evaluate> --finetuned-model-name <model_name>
```
where ```<task_to_evaluate>``` can be one of ```human_eval```, ```alpaca_eval``` ot ```MATH```.

In the case of ```human_eval```, it is further necessary to run 
```
evaluate_functional_correctness <created_json_file>
```
where ```<created_json_file>``` is the output file created using the previous command.

In the case of ```alpaca_eval```, it is further necessary to run

```
alpaca_eval --model_outputs <outputs_path> --annotators_config chatgpt_fn --name <model_name>
```
where ```<created_json_file>``` is the file output created using the previous command.

To run the evaluation of GSM8k and SciQ, please follow the [lm-evaluation-harness repository](https://github.com/EleutherAI/lm-evaluation-harness).

## Acknowledgements
The current repository was built on top of the repository of the paper [Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch](https://github.com/yule-BUAA/MergeLM/blob/main/README.md). We are grateful to the authors for making their project and code publicly available.