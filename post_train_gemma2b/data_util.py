from datasets import load_dataset, Dataset


def load_data_ultra_feedback() -> Dataset:
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized",split='train_prefs',cache_dir="S:/tmp/jing/gemma2b/hf_cache")
    print(ds)
    prompts = []
    for row in ds:
        prompt = row['prompt']
        chosen = row['chosen'][1]['content']
        rejected = row['rejected'][1]['content']
        prompts.append({'prompt': prompt, 'chosen': chosen, 'rejected': rejected})
    return Dataset.from_list(prompts)

ds = load_data_ultra_feedback()
print(ds[0])
# print('------------------------')
# print(ds[0]['rejected'])


