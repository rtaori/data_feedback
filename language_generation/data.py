import numpy as np
from collections import defaultdict
from datasets import load_dataset


# Load and tokenize dataset
def load_rtp_dataset(data_path, tokenizer):
    dataset = load_dataset('json', data_files=data_path, split='all')

    # isolate out only the prompt and continuation text
    dataset = dataset.map(lambda prompt, continuation: {'prompt': prompt['text'], 'continuation': continuation['text']},
                        input_columns=['prompt', 'continuation'], remove_columns=list(dataset.features.keys()))

    def tokenize_rtp(prompt, continuation):
        # tokenize prompt and continuation separately
        prompt_ids = tokenizer(prompt)['input_ids']
        continuation_ids = tokenizer(continuation)['input_ids']
        return {'prompt_ids': prompt_ids, 'continuation_ids': continuation_ids}

    dataset = dataset.map(tokenize_rtp, input_columns=['prompt', 'continuation'])
    return dataset

# tokenize for training or evaluation
def tokenize_training(prompt_ids, continuation_ids):
    input_ids = prompt_ids + continuation_ids
    attention_mask = [1] * (len(prompt_ids) + len(continuation_ids))
    # calculate labels for training to ignore loss calculation for prompt portion (only calculate on the continuation) 
    labels = [-100] * len(prompt_ids) + continuation_ids
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

# tokenizer for generation (only need prompts)
def tokenize_generation(prompt_ids):
    return {'input_ids': prompt_ids, 'attention_mask': [1] * len(prompt_ids)}

def prep_dataset_for_training(dataset):
    return dataset.map(tokenize_training, input_columns=['prompt_ids', 'continuation_ids'], 
                       remove_columns=list(dataset.features.keys()), keep_in_memory=True)

def prep_dataset_for_generation(dataset):
    return dataset.map(tokenize_generation, input_columns=['prompt_ids'], 
                       remove_columns=list(dataset.features.keys()), keep_in_memory=True)


# Randomly split dataset into two determined by partition_size
def rand_split_dataset(dataset, partition_size):
    if partition_size == len(dataset):
        return dataset, []
    dataset_split = dataset.train_test_split(test_size=partition_size, shuffle=True)
    return dataset_split['test'], dataset_split['train']


def get_ngram_overlap(train_tokens, output_tokens, n):
    train_ngrams = [tuple(train_token[i:i+n]) for train_token in train_tokens for i in range(len(train_token)-n+1)]
    output_ngrams = [tuple(output_token[i:i+n]) for output_token in output_tokens for i in range(len(output_token)-n+1)]
    # construct train n_grams
    train_counts = defaultdict(int)
    for n_gram in train_ngrams:
        train_counts[n_gram] += 1
    # for each output n_gram, find how many times it appeared in the train set
    output_ngram_freq = []
    for n_gram in output_ngrams:
        output_ngram_freq.append(train_counts[n_gram])
    # return fraction of output n_grams that are in train set
    return (np.array(output_ngram_freq) != 0).mean()
