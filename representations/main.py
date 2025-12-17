import datasets
import numpy as np
import torch
from measure_alignment import *
from category import FormatCategory
import pickle

def compute_alignment(x_loaders, y_loaders, metric, topk, precise=True, output_dir="results/alignment", layer: int | list[int] | None = None, filter_function = lambda x: True, random_N = None, topk_multiplier = None, dataset = False, cut_len = None, sample_indices = None, x_shape = None):
    """
    Modified from the repo [platonic representation hypothesis]

    Args:
        x_feat_paths: list of paths to x features
        y_feat_paths: list of paths to y features
        metric: the metric to use
        topk: the number of nearest neighbors to use (specific to knn metrics), really topk * N, topk here is the multiplier
        precise: if true use exact quantiling. (helpful to set to false if running on cpu)
            this is more of a feature to speed up matmul if using float32 
            used in measure_alignment.py
    Returns:
        alignment_scores: a numpy array of shape len(x_feat_paths) x len(y_feat_paths)
        alignment_indices: a numpy array of shape len(x_feat_paths) x len(y_feat_paths) x 2
    """
    
    os.makedirs(output_dir, exist_ok=True)
    assert not (random_N and cut_len), "random_N and cut_len cannot be used together"
    symmetric_metric = (x_loaders == y_loaders)
    if metric == "cycle_knn":
        symmetric_metric = False

    if 'raw' not in metric:
        alignment_scores = np.zeros((len(x_loaders), len(y_loaders)))
        alignment_indices = np.zeros((len(x_loaders), len(y_loaders), 2))
    else:   
        assert x_shape is not None, "x_shape must be provided for raw metrics"
        alignment_scores = np.zeros((len(x_loaders), len(y_loaders), x_shape), dtype=object)
        alignment_indices = np.zeros((len(x_loaders), len(y_loaders), x_shape, 2), dtype=object)

    pbar = tqdm(total=len(y_loaders) * len(x_loaders))

    for i, x_loader in enumerate(x_loaders):
        # raw_x = torch.load(x_fp, map_location="cuda:0")["feats"]
        raw_x = x_loader.load_features(layer=layer, filter_function=filter_function, dataset=dataset, sample_indices=sample_indices)
        if isinstance(raw_x, torch.Tensor):
            x_feats = prepare_features(raw_x.float(), exact=precise)

        else:
            x_feats = [prepare_features(layer.float(), exact=precise) for layer in raw_x]
        
        # x_feats = prepare_features(torch.load(x_fp, map_location="cuda:0")["feats"].float(), exact=precise)

        for j, y_loader in enumerate(y_loaders):
            if symmetric_metric:
                if i > j:
                    pbar.update(1)
                    continue           

            # raw_y = torch.load(y_fp, map_location="cuda:0")["feats"]
            raw_y = y_loader.load_features(layer=layer, filter_function=filter_function, dataset=dataset, sample_indices=sample_indices)
            if isinstance(raw_y, torch.Tensor):
                y_feats = prepare_features(raw_y.float(), exact=precise)
            else:
                y_feats = [prepare_features(layer.float(), exact=precise) for layer in raw_y]

            if random_N is not None:
                # ensure same random selection
                np.random.seed(42)
                if isinstance(x_feats, list):
                    rand_indices = np.random.choice(x_feats[0].shape[0], size=random_N, replace=False)
                    x_feats = [x_feat[rand_indices] for x_feat in x_feats]
                    y_feats = [y_feat[rand_indices] for y_feat in y_feats]
                else:
                    rand_indices = np.random.choice(x_feats.shape[0], size=random_N, replace=False)
                    x_feats = x_feats[rand_indices]
                    y_feats = y_feats[rand_indices]
            if cut_len is not None:
                if isinstance(x_feats, list):
                    x_feats = [x_feat[:cut_len] for x_feat in x_feats]
                    y_feats = [y_feat[:cut_len] for y_feat in y_feats]
                else:
                    x_feats = x_feats[:cut_len]
                    y_feats = y_feats[:cut_len]
            if topk_multiplier is not None:
                best_score, best_indices = compute_score(y_feats, x_feats, metric=metric, topk=int(topk_multiplier * x_feats.shape[0]), x_shape=x_shape)
            else:
                best_score, best_indices = compute_score(y_feats, x_feats, metric=metric, topk=topk, x_shape=x_shape)

            alignment_scores[i, j] = best_score
            alignment_indices[i, j] = best_indices
            
            if symmetric_metric:
                alignment_scores[j, i] = best_score
                alignment_indices[j, i] = best_indices[::-1]

            pbar.update(1)

            del y_feats
            torch.cuda.empty_cache()
    
    if 'raw' in metric:
        alignment_scores = alignment_scores.swapaxes(0, 2)

    return alignment_scores, alignment_indices

class FeaturesLoader:
    DATASET_NAME = "SHENJJ1017/poem_aesthetic_eval"
    DATASET_NAME = 'SHENJJ1017/Image-Text'
    REVISION = "main"
    # REVISION = "aligned_s"
    DATASET = datasets.load_dataset(DATASET_NAME, revision=REVISION, split='train')

    def __init__(self, features_path):
        self.features_path = features_path
        self.features = None

    def __str__(self):
        return f"FeaturesLoader(features_path={self.features_path})"
    
    def __repr__(self):
        return self.__str__()
    
    def _load_file(self, numpy = False):
        if numpy:
            return torch.load(self.features_path)['feats'].float().numpy()
        return torch.load(self.features_path)['feats']

    def _load_layer_features(self, features, layer: int | list[int] | None = -1) -> np.ndarray:
        """
        Load precomputed radius features from a .pt file.

        Args:
            features_path: Path to the .pt file containing the features.

        Returns:
            A NumPy array containing the loaded features.
        """
        if isinstance(layer, list):
            return np.array([features[:, l, :] for l in layer])
        elif layer is None:
            return features
        elif isinstance(layer, int):
            return features[:, layer, :]
        else:
            raise ValueError(f"Unsupported layer type: {type(layer)}")

    def load_features(self, layer: int | list[int] | None = -1, filter_function = lambda x: True, dataset = False, numpy = False, sample_indices = None) -> np.ndarray:
        if dataset:
            bool_list = [filter_function(item) for item in self.DATASET] if filter_function is not None else [True] * len(self.DATASET)
        features = self._load_file(numpy=numpy) # N * Layer * D
        if sample_indices is not None:
            features = features[sample_indices]
            return self._load_layer_features(features, layer=layer)
        if not dataset:
            bool_list = [True] * features.shape[0]
        return self._load_layer_features(features, layer=layer)[bool_list]

    def length_distribution(self):
        lens = [len(dp['content'].split()) for dp in self.DATASET]
        return lens

def get_filter_pair_function(filter_name: str):
    if filter_name == "aesthetic_score":
        return lambda x: x['score'] <=3, lambda x: x['score'] >=4
    if filter_name == "form":
        return lambda x: x['form'] in FormatCategory.NONFORMATED, lambda x: x['form'] in FormatCategory.FORMATED
    if filter_name == "length":
        return lambda x: len(x['content'].split()) <=83, lambda x: len(x['content'].split()) >83


if __name__ == "__main__":


    text_type = "features/image-texts"
    loaders_poemes = [
        FeaturesLoader(f"../platonic/{text_type}/NousResearch_Meta-Llama-3-8B_pool-avg.pt"),
        FeaturesLoader(f"../platonic/{text_type}/mistralai_Mistral-7B-v0.1_pool-avg.pt"),
        FeaturesLoader(f"../platonic/{text_type}/google_gemma-7b_pool-avg.pt"),
        FeaturesLoader(f"../platonic/{text_type}/google_gemma-2b_pool-avg.pt"),
        FeaturesLoader(f"../platonic/{text_type}/bigscience_bloomz-1b7_pool-avg.pt"),
        FeaturesLoader(f"../platonic/{text_type}/allenai_OLMo-1B-hf_pool-avg.pt"),
        FeaturesLoader(f"../platonic/{text_type}/bigscience_bloomz-560m_pool-avg.pt"),
    ]

    text_type = "features/text/wit_1024"
    loaders_text = [
        FeaturesLoader(f"../platonic/{text_type}/NousResearch_Meta-Llama-3-8B_pool-avg.pt"),
        FeaturesLoader(f"../platonic/{text_type}/mistralai_Mistral-7B-v0.1_pool-avg.pt"),
        FeaturesLoader(f"../platonic/{text_type}/google_gemma-7b_pool-avg.pt"),
        FeaturesLoader(f"../platonic/{text_type}/google_gemma-2b_pool-avg.pt"),
        FeaturesLoader(f"../platonic/{text_type}/bigscience_bloomz-1b7_pool-avg.pt"),
        FeaturesLoader(f"../platonic/{text_type}/allenai_OLMo-1B-hf_pool-avg.pt"),
        FeaturesLoader(f"../platonic/{text_type}/bigscience_bloomz-560m_pool-avg.pt"),
    ]

    image_loaders = [
        FeaturesLoader(f"../platonic/features/images/vit_huge_patch14_clip_224.laion2b_ft_in12k_pool-cls.pt"),
        FeaturesLoader(f"../platonic/features/images/vit_large_patch14_clip_224.laion2b_pool-cls.pt"),
        FeaturesLoader(f"../platonic/features/images/vit_large_patch14_dinov2.lvd142m_pool-cls.pt"),
        FeaturesLoader(f"../platonic/features/images/vit_base_patch16_224.mae_pool-cls.pt"),
        FeaturesLoader(f"../platonic/features/images/vit_base_patch14_dinov2.lvd142m_pool-cls.pt"),
        FeaturesLoader(f"../platonic/features/images/vit_small_patch16_224.augreg_in21k_pool-cls.pt"),
        FeaturesLoader(f"../platonic/features/images/vit_tiny_patch16_224.augreg_in21k_pool-cls.pt"),
    ]
    
    poems = datasets.load_dataset('SHENJJ1017/poem_aesthetic_eval', revision="main", split='train')
    texts = datasets.load_dataset('minhuh/prh', revision="wit_1024")
    cut_len = min(len(poems), len(texts)) # ensure fair comparison
    cut_len = 993

    alignment_scores, alignment_indices = compute_alignment(loaders_poemes, image_loaders, "mutual_knn_raw", 10, False, dataset=False, cut_len=cut_len, x_shape=cut_len)
    # save npy arrays
    np.save("texts_images_raw_mutual_knn_scores.npy", alignment_scores)
    
    # N * 7 * 7 -> 7 * 7 by mean
    mean_scores = np.mean(alignment_scores, axis=0)
    print("Mean scores between text features and image features:")
    print(mean_scores)
    raise ValueError("Stop here")
    # Poem alignment 
    alignment_scores, alignment_indices = compute_alignment(loaders_poemes, loaders_poemes, "mutual_knn", 10, False, dataset=True, cut_len=cut_len)

    # Text alignment
    alignment_scores, alignment_indices = compute_alignment(loaders_text, loaders_text, "mutual_knn", 10, False, dataset=False, cut_len=cut_len)

    # example of alignment with filters
    filter_1, filter_2 = get_filter_pair_function("length")
    # short poemss
    alignment_scores, alignment_indices = compute_alignment(loaders_poemes, loaders_poemes, "mutual_knn", 10, False, dataset=True, filter_function=filter_1, cut_len=cut_len)
    # long poemss
    alignment_scores, alignment_indices = compute_alignment(loaders_poemes, loaders_poemes, "mutual_knn", 10, False, dataset=True, filter_function=filter_2, cut_len=cut_len)