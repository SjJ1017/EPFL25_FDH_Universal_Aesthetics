import numpy as np

fp = '/Users/shenjiajun/Desktop/EPFL/Courses/FDH/Final Project/Codes/platonic/results/alignment/Ozziey/poems_dataset/local_test/language_pool-avg_prompt-False_language_pool-avg_prompt-False/mutual_knn_k10.npy'
result = np.load(fp, allow_pickle=True).item()
print(result.keys())
print(result['scores'].shape)
print(result)