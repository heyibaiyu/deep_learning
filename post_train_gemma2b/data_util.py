from datasets import load_dataset, Dataset
import numpy as np # Import numpy for statistical calculations


def load_data_ultra_feedback(flag='train') -> Dataset:
    if flag == 'train':
        ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized",split='train_prefs',cache_dir="S:/tmp/jing/gemma2b/hf_cache")
    else:
        ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized",split='test_prefs',cache_dir="S:/tmp/jing/gemma2b/hf_cache_test")
    prompts = []
    empty_response = 0
    for row in ds:
        prompt = row['prompt']
        chosen = row['chosen'][1]['content']
        rejected = row['rejected'][1]['content']
        # Add a check to skip if either chosen or rejected is empty
        if chosen and rejected: # This checks if the strings are not empty
            prompts.append({'prompt': prompt, 'chosen': chosen, 'rejected': rejected})
        else:
            # print(f"Skipping example due to empty chosen or rejected response. Prompt: {prompt[:50]}...")
            empty_response += 1
    print(f'Removed {empty_response} empty responses')
    print(prompts[0].keys())
    return Dataset.from_list(prompts)

def analyze_response_lengths(dataset: Dataset, dataset_name: str):
    chosen_lengths = [len(item['chosen'].split()) for item in dataset]
    rejected_lengths = [len(item['rejected'].split()) for item in dataset]

    avg_chosen_length = sum(chosen_lengths) / len(chosen_lengths)
    avg_rejected_length = sum(rejected_lengths) / len(rejected_lengths)

    print(f"\n--- {dataset_name} Dataset Response Length Analysis ---")
    print(f"Average chosen response length: {avg_chosen_length:.2f} words")
    print(f"Average rejected response length: {avg_rejected_length:.2f} words")

    if avg_chosen_length > avg_rejected_length:
        print(f"Chosen responses are, on average, {avg_chosen_length - avg_rejected_length:.2f} words longer than rejected responses.")
    elif avg_rejected_length > avg_chosen_length:
        print(f"Rejected responses are, on average, {avg_rejected_length - avg_chosen_length:.2f} words longer than chosen responses.")
    else:
        print("Average chosen and rejected response lengths are equal.")

def analyze_length_ratios(dataset: Dataset, dataset_name: str):
    ratios = []
    for item in dataset:
        len_chosen = len(item['chosen'].split())
        len_rejected = len(item['rejected'].split())
        if len_rejected > 0:
            ratios.append(len_chosen / len_rejected)

        else:
            # Handle cases where rejected response is empty
            ratios.append(float('inf') if len_chosen > 0 else 1.0) # If chosen is also empty, ratio is 1.0

    ratios = np.array(ratios)

    print(f"\n--- {dataset_name} Dataset Length Ratio (Chosen/Rejected) Analysis ---")
    print(f"Min ratio: {np.min(ratios):.2f}")
    print(f"Max ratio: {np.max(ratios):.2f}")
    print(f"Mean ratio: {np.mean(ratios):.2f}")
    print(f"Median ratio: {np.median(ratios):.2f}")
    print(f"25th percentile ratio: {np.percentile(ratios, 25):.2f}")
    print(f"75th percentile ratio: {np.percentile(ratios, 75):.2f}")
    print(f"90th percentile ratio: {np.percentile(ratios, 90):.2f}")
    # print(f"Number of infinite ratios (rejected response was empty): {np.isinf(ratios).sum()}")

def analyze_length_comparison_distribution(dataset: Dataset, dataset_name: str):
    chosen_longer = 0
    rejected_longer = 0
    equal_length = 0
    total_examples = len(dataset)

    for item in dataset:
        len_chosen = len(item['chosen'].split())
        len_rejected = len(item['rejected'].split())

        if len_chosen > len_rejected:
            chosen_longer += 1
        elif len_rejected > len_chosen:
            rejected_longer += 1
        else:
            equal_length += 1

    print(f"\n--- {dataset_name} Dataset Length Comparison Distribution (Chosen vs Rejected) ---")
    print(f"Chosen is longer: {chosen_longer} ({chosen_longer / total_examples:.2%})")
    print(f"Rejected is longer: {rejected_longer} ({rejected_longer / total_examples:.2%})")
    print(f"Lengths are equal: {equal_length} ({equal_length / total_examples:.2%})")


# Load and analyze training data
train_dataset = load_data_ultra_feedback('train')
analyze_response_lengths(train_dataset, 'Training')
analyze_length_ratios(train_dataset, 'Training')
analyze_length_comparison_distribution(train_dataset, 'Training')

# Load and analyze test data
test_dataset = load_data_ultra_feedback('test')
analyze_response_lengths(test_dataset, 'Test')
analyze_length_ratios(test_dataset, 'Test')
analyze_length_comparison_distribution(test_dataset, 'Test')


# train = load_data_ultra_feedback()
# print('data example', train[0])
# print('test data shape', train.shape)

# data example {'prompt': 'how can i develop a habit of drawing daily', 'chosen': "Developing a daily habit of drawing can be challenging but with consistent practice and a few tips, it can become an enjoyable and rewarding part of your daily routine. Here are some strategies to help you develop the habit of drawing daily:\n\n1. Set a specific time: Allocate a specific time of the day to draw. It could be in the morning, afternoon, or evening. Make drawing a part of your daily routine.\n2. Set a specific duration: Determine the amount of time you want to spend on drawing each day. It can be as little as 10 minutes or as long as an hour. Be consistent with the duration to help build the habit.\n3. Start small and simple: Don't try to create a masterpiece every day, start with simple and easy-to-do sketches. Focus on improving your skills gradually.\n4. Use a variety of tools and mediums: Experiment with different tools like pencils, pens, markers, and different mediums like paper, canvas, or digital apps to keep your drawing practice interesting and engaging.\n5. Take breaks and rest: Taking breaks and resting after some time of drawing can help you avoid burnout and stay motivated.\n6. Challenge yourself: Set challenges like drawing objects from memory or a specific subject to improve your skills and keep your drawing practice interesting.\n7. Track your progress: Keep a record of your daily drawing practice and track your progress. This can be a source of motivation and help you see how far you've come.\n\nRemember, developing a habit takes time and patience. Stay consistent with your drawing practice, be flexible and open to trying new things, and with time, you'll develop a habit of daily drawing that brings you joy and satisfaction.", 'rejected': "As an AI language model, I cannot personally develop habits for you. But, here are some tips for developing a habit of drawing daily:\n\n1. Start small: Start with simple drawings or doodles and gradually increase the complexity of your drawings.\n\n2. Set a schedule: Set a specific time every day to draw, and try to stick to it as much as possible.\n\n3. Make it fun: Don't pressure yourself to create masterpieces every time you draw. Make it a relaxing and enjoyable experience.\n\n4. Use resources: There are many drawing tutorials available online. Use resources like YouTube or online drawing courses to help you improve your skills.\n\n5. Surround yourself with inspiration: Expose yourself to a variety of art forms, such as paintings, illustrations, and photographs, to inspire and motivate you.\n\nRemember, everyone has their own creative style and pace. Just keep practicing and enjoying the process of drawing."}
# training data shape (61135, 3)
# test data shape (2000, 3)

#
# test = load_data_ultra_feedback('test')
# print('training data shape', test.shape)

__all__ = ['load_data_ultra_feedback']