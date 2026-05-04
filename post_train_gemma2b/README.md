# PreferenceLab: DPO Post-Training Gemma-2B

## TL;DR
Fine-tuned Gemma-2B-IT with DPO on UltraFeedback preference data.
Compared base vs DPO checkpoints using reward-margin win rate, AlpacaEval, LM-Eval Harness, perplexity, and manual side-by-side evaluation.

## Why this project
Modern LLM post-training is fundamentally about improving model behavior after pretraining or instruction tuning. Preference optimization methods such as DPO provide a practical way to align a model with human judgments by converting each prompt into a pairwise ranking problem: the preferred response should become more likely under the policy than the rejected response, while staying close to the reference model.

This project implements a compact post-training workflow on Gemma-2B using UltraFeedback preference data, LoRA-based DPO training, implicit reward-margin evaluation, benchmark evaluation, and qualitative failure analysis. The goal is not just to fine-tune a model, but to study how preference optimization changes response quality, reward margins, benchmark performance, and failure modes.

## Model and data
- Base policy: google/gemma-2b-it
- Preference data: UltraFeedback binarized
    * UltraFeedback dataset: general-purpose preference data; diverse topic
    https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized/viewer/default/train_prefs?views%5B%5D=train_prefs
    * Goal: human preference
- Train: 61,135 examples
- Test: 2,000 examples
- Training method: LoRA + 4-bit quantization

## Training setup
- LoRA rank: 16
- beta: 0.1
- max steps: 7,000
- batch size: 2
- grad accumulation: 4
- max length: 512
- GPU: RTX 3090
- training time: 9h


### DPO training result
[dpo_train_human_preference.py](dpo_train_human_preference.py)

![img_2.png](img_2.png)

### DPO Evaluation on human preference

Metrics: Win Rate + Reward Margin on held-out preference pairs (`test_prefs`)

Script: [eval_win_rate.py](eval_win_rate.py)

Run:
```bash
python3 eval_win_rate.py \
  --adapter-path checkpoints/checkpoint-7000 \
  --base-model google/gemma-2b-it \
  --batch-size 4 \
  --max-length 1024 \
  --output-json eval_out/dpo_winrate.json
```

The script prints JSON metrics:
* `win_rate`: fraction where DPO reward(chosen) > DPO reward(rejected)
* `avg_margin`: mean reward(chosen) - reward(rejected)
* `avg_reward_chosen` / `avg_reward_rejected`
* `evaluated_examples` / `skipped_examples`

![img_1.png](img_1.png)

### Perplexity 
output: [perplexity_out](eval_out/perplexity_out) 

script: [eval_perplexity.py](eval_perplexity.py)

## Alpaca-eval 
    python3 run_alpacaeval_local.py

Dataset: 
    https://huggingface.co/datasets/tatsu-lab/alpaca_eval/blob/main/alpaca_eval.json


## Human side by side eval
output: [manual_eval_candidates.json](manual_eval_candidates.json)

script: [human_eval.py](human_eval.py), 


## LM-Eval Harness (MMLU, TruthfulQA, etc.) 
This evaluates objective quality, but not human preference.

|                 Tasks                 |Version|Filter|n-shot|  Metric   |   | Value |   |Stderr|
|---------------------------------------|------:|------|-----:|-----------|---|------:|---|-----:|
|hellaswag                              |      1|none  |     0|acc        |↑  | 0.4922|±  |0.0050|
|                                       |       |none  |     0|acc_norm   |↑  | 0.6443|±  |0.0048|
|mmlu                                   |      2|none  |      |acc        |↑  | 0.3816|±  |0.0040|
| - humanities                          |      2|none  |      |acc        |↑  | 0.3552|±  |0.0069|
|  - formal_logic                       |      1|none  |     0|acc        |↑  | 0.3571|±  |0.0429|
|  - high_school_european_history       |      1|none  |     0|acc        |↑  | 0.5212|±  |0.0390|
|  - high_school_us_history             |      1|none  |     0|acc        |↑  | 0.4167|±  |0.0346|
|  - high_school_world_history          |      1|none  |     0|acc        |↑  | 0.5443|±  |0.0324|
|  - international_law                  |      1|none  |     0|acc        |↑  | 0.5207|±  |0.0456|
|  - jurisprudence                      |      1|none  |     0|acc        |↑  | 0.4907|±  |0.0483|
|  - logical_fallacies                  |      1|none  |     0|acc        |↑  | 0.3742|±  |0.0380|
|  - moral_disputes                     |      1|none  |     0|acc        |↑  | 0.4046|±  |0.0264|
|  - moral_scenarios                    |      1|none  |     0|acc        |↑  | 0.2413|±  |0.0143|
|  - philosophy                         |      1|none  |     0|acc        |↑  | 0.3505|±  |0.0271|
|  - prehistory                         |      1|none  |     0|acc        |↑  | 0.3858|±  |0.0271|
|  - professional_law                   |      1|none  |     0|acc        |↑  | 0.3129|±  |0.0118|
|  - world_religions                    |      1|none  |     0|acc        |↑  | 0.4620|±  |0.0382|
| - other                               |      2|none  |      |acc        |↑  | 0.4287|±  |0.0088|
|  - business_ethics                    |      1|none  |     0|acc        |↑  | 0.4400|±  |0.0499|
|  - clinical_knowledge                 |      1|none  |     0|acc        |↑  | 0.4113|±  |0.0303|
|  - college_medicine                   |      1|none  |     0|acc        |↑  | 0.3584|±  |0.0366|
|  - global_facts                       |      1|none  |     0|acc        |↑  | 0.2300|±  |0.0423|
|  - human_aging                        |      1|none  |     0|acc        |↑  | 0.4484|±  |0.0334|
|  - management                         |      1|none  |     0|acc        |↑  | 0.4854|±  |0.0495|
|  - marketing                          |      1|none  |     0|acc        |↑  | 0.5983|±  |0.0321|
|  - medical_genetics                   |      1|none  |     0|acc        |↑  | 0.3900|±  |0.0490|
|  - miscellaneous                      |      1|none  |     0|acc        |↑  | 0.4943|±  |0.0179|
|  - nutrition                          |      1|none  |     0|acc        |↑  | 0.4673|±  |0.0286|
|  - professional_accounting            |      1|none  |     0|acc        |↑  | 0.3227|±  |0.0279|
|  - professional_medicine              |      1|none  |     0|acc        |↑  | 0.3051|±  |0.0280|
|  - virology                           |      1|none  |     0|acc        |↑  | 0.3675|±  |0.0375|
| - social sciences                     |      2|none  |      |acc        |↑  | 0.4274|±  |0.0088|
|  - econometrics                       |      1|none  |     0|acc        |↑  | 0.2544|±  |0.0410|
|  - high_school_geography              |      1|none  |     0|acc        |↑  | 0.4495|±  |0.0354|
|  - high_school_government_and_politics|      1|none  |     0|acc        |↑  | 0.5026|±  |0.0361|
|  - high_school_macroeconomics         |      1|none  |     0|acc        |↑  | 0.3795|±  |0.0246|
|  - high_school_microeconomics         |      1|none  |     0|acc        |↑  | 0.3361|±  |0.0307|
|  - high_school_psychology             |      1|none  |     0|acc        |↑  | 0.5174|±  |0.0214|
|  - human_sexuality                    |      1|none  |     0|acc        |↑  | 0.4046|±  |0.0430|
|  - professional_psychology            |      1|none  |     0|acc        |↑  | 0.3824|±  |0.0197|
|  - public_relations                   |      1|none  |     0|acc        |↑  | 0.4000|±  |0.0469|
|  - security_studies                   |      1|none  |     0|acc        |↑  | 0.3837|±  |0.0311|
|  - sociology                          |      1|none  |     0|acc        |↑  | 0.5373|±  |0.0353|
|  - us_foreign_policy                  |      1|none  |     0|acc        |↑  | 0.5700|±  |0.0498|
| - stem                                |      2|none  |      |acc        |↑  | 0.3298|±  |0.0082|
|  - abstract_algebra                   |      1|none  |     0|acc        |↑  | 0.3000|±  |0.0461|
|  - anatomy                            |      1|none  |     0|acc        |↑  | 0.4074|±  |0.0424|
|  - astronomy                          |      1|none  |     0|acc        |↑  | 0.3750|±  |0.0394|
|  - college_biology                    |      1|none  |     0|acc        |↑  | 0.4514|±  |0.0416|
|  - college_chemistry                  |      1|none  |     0|acc        |↑  | 0.3100|±  |0.0465|
|  - college_computer_science           |      1|none  |     0|acc        |↑  | 0.3600|±  |0.0482|
|  - college_mathematics                |      1|none  |     0|acc        |↑  | 0.3200|±  |0.0469|
|  - college_physics                    |      1|none  |     0|acc        |↑  | 0.1863|±  |0.0387|
|  - computer_security                  |      1|none  |     0|acc        |↑  | 0.5000|±  |0.0503|
|  - conceptual_physics                 |      1|none  |     0|acc        |↑  | 0.3362|±  |0.0309|
|  - electrical_engineering             |      1|none  |     0|acc        |↑  | 0.4966|±  |0.0417|
|  - elementary_mathematics             |      1|none  |     0|acc        |↑  | 0.2460|±  |0.0222|
|  - high_school_biology                |      1|none  |     0|acc        |↑  | 0.4516|±  |0.0283|
|  - high_school_chemistry              |      1|none  |     0|acc        |↑  | 0.2808|±  |0.0316|
|  - high_school_computer_science       |      1|none  |     0|acc        |↑  | 0.3700|±  |0.0485|
|  - high_school_mathematics            |      1|none  |     0|acc        |↑  | 0.2185|±  |0.0252|
|  - high_school_physics                |      1|none  |     0|acc        |↑  | 0.2848|±  |0.0368|
|  - high_school_statistics             |      1|none  |     0|acc        |↑  | 0.2269|±  |0.0286|
|  - machine_learning                   |      1|none  |     0|acc        |↑  | 0.3214|±  |0.0443|
|truthfulqa_gen                         |      3|none  |     0|bleu_acc   |↑  | 0.4088|±  |0.0172|
|                                       |       |none  |     0|bleu_diff  |↑  |-2.0255|±  |0.5568|
|                                       |       |none  |     0|bleu_max   |↑  |18.5645|±  |0.6768|
|                                       |       |none  |     0|rouge1_acc |↑  | 0.4113|±  |0.0172|
|                                       |       |none  |     0|rouge1_diff|↑  |-2.8608|±  |0.7197|
|                                       |       |none  |     0|rouge1_max |↑  |41.2027|±  |0.8277|
|                                       |       |none  |     0|rouge2_acc |↑  | 0.2876|±  |0.0158|
|                                       |       |none  |     0|rouge2_diff|↑  |-4.2040|±  |0.8030|
|                                       |       |none  |     0|rouge2_max |↑  |24.9820|±  |0.8908|
|                                       |       |none  |     0|rougeL_acc |↑  | 0.3868|±  |0.0170|
|                                       |       |none  |     0|rougeL_diff|↑  |-3.2768|±  |0.7242|
|                                       |       |none  |     0|rougeL_max |↑  |38.4656|±  |0.8277|
|truthfulqa_mc1                         |      2|none  |     0|acc        |↑  | 0.2889|±  |0.0159|
|truthfulqa_mc2                         |      3|none  |     0|acc        |↑  | 0.4576|±  |0.0159|                       |      3|none  |     0|acc        |↑  | 0.4576|±  |0.0159|

|      Groups      |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|------------------|------:|------|------|------|---|-----:|---|-----:|
|mmlu              |      2|none  |      |acc   |↑  |0.3816|±  |0.0040|
| - humanities     |      2|none  |      |acc   |↑  |0.3552|±  |0.0069|
| - other          |      2|none  |      |acc   |↑  |0.4287|±  |0.0088|
| - social sciences|      2|none  |      |acc   |↑  |0.4274|±  |0.0088|
| - stem           |      2|none  |      |acc   |↑  |0.3298|±  |0.0082|


## Analysis
- Length bias
  ### Length imbalance in preference data

Before interpreting DPO win rate, I checked whether the preference dataset has a length bias. This matters because preference datasets often favor longer, more detailed answers, and DPO may learn to prefer verbosity if response length is correlated with the `chosen` label.

| Split | Avg chosen length | Avg rejected length | Avg difference | Relative difference |
|---|---:|---:|---:|---:|
| Train | 198.34 words | 175.48 words | +22.86 words | +13.0% |
| Test | 204.65 words | 176.91 words | +27.74 words | +15.7% |

Chosen is longer: 1086 (54.38%)
Rejected is longer: 872 (43.67%)
Lengths are equal: 39 (1.95%)

Both splits show that `chosen` responses are longer than `rejected` responses on average. In the training set, chosen responses are **22.86 words longer** than rejected responses, a relative increase of approximately **13.0%**. In the held-out test set, chosen responses are **27.74 words longer**, a relative increase of approximately **15.7%**.

This suggests that the preference labels are partially correlated with response length. 

According to the ratio analysis when chosen is longer vs. rejected is longer, the distribution is actually good. The high percentage of cases where the shorter answer wins (44%) provides enough "negative length signal" to prevent the model from learning a simple "longer is better" heuristic.

- Failure cases
- Limitations

