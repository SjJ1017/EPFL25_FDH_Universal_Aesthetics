from datasets import load_dataset
dataset =  load_dataset('minhuh/prh')
print(dataset['train'][0]['text'])  # Print the first example in the training set

texts = []
for i in range(len(dataset['train'])):
    texts.append(dataset['train'][i]['text'])

import random
sampled_texts = random.sample(texts, 10)
for text in sampled_texts:
    print("-----")
    print(text)