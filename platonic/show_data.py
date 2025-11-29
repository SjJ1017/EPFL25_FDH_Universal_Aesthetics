from datasets import load_dataset

dataset = 'minhuh/prh'
data = load_dataset(dataset, revision='wit_1024', split='train')
print(data.info.homepage)