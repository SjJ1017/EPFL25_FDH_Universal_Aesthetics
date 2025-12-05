from datasets import load_dataset
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')



def word_cloud(words, file_path=None):
    # Generate a word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(words))

    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    if file_path:
        plt.savefig(file_path, dpi=300)
    else:
        plt.savefig("poem_wordcloud_tmp.png", dpi=300)
    plt.show()


class SemanticEncoder:
    """
    Semantic encoder using encoder-only transformer models.
    Supports multiple encoder architectures.
    """
    
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', device=None):
        """
        Initialize the semantic encoder.
        
        Args:
            model_name: Name of the pretrained model. Options:
                - 'sentence-transformers/all-MiniLM-L6-v2' (default, 384 dim, fast)
                - 'sentence-transformers/all-mpnet-base-v2' (768 dim, high quality)
                - 'bert-base-uncased' (768 dim)
                - 'roberta-base' (768 dim)
                - 'microsoft/deberta-v3-base' (768 dim)
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading model: {model_name}")
        print(f"Using device: {self.device}")
        
        # Use sentence-transformers if available (optimized for semantic similarity)
        if 'sentence-transformers' in model_name or model_name.startswith('all-'):
            self.use_sentence_transformer = True
            self.model = SentenceTransformer(model_name, device=self.device)
        else:
            self.use_sentence_transformer = False
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
        
        print("Model loaded successfully!")
    
    def encode_sentence(self, sentence, max_length=512, pooling='mean'):
        """
        Encode a single sentence into a semantic vector.
        
        Args:
            sentence: Input text string
            max_length: Maximum sequence length
            pooling: Pooling strategy ('mean', 'cls', 'max')
                - 'mean': Average of all token embeddings (recommended)
                - 'cls': Use [CLS] token embedding
                - 'max': Max pooling over token embeddings
        
        Returns:
            numpy array: Semantic vector
        """
        if self.use_sentence_transformer:
            # Sentence-BERT handles everything internally
            with torch.no_grad():
                embedding = self.model.encode(sentence, convert_to_numpy=True, show_progress_bar=False)
            return embedding
        
        # Manual encoding for standard transformers
        with torch.no_grad():
            # Tokenize
            inputs = self.tokenizer(
                sentence, 
                return_tensors='pt', 
                truncation=True, 
                max_length=max_length,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model outputs
            outputs = self.model(**inputs)
            
            # Extract embeddings
            if pooling == 'cls':
                # Use [CLS] token (first token)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            elif pooling == 'max':
                # Max pooling
                embedding = torch.max(outputs.last_hidden_state, dim=1)[0].cpu().numpy()[0]
            else:  # mean pooling (default)
                # Mean pooling with attention mask
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embedding = (sum_embeddings / sum_mask).cpu().numpy()[0]
        
        return embedding
    
    def encode_batch(self, sentences, batch_size=32, max_length=512, pooling='mean', show_progress=True):
        """
        Encode multiple sentences into semantic vectors.
        
        Args:
            sentences: List of text strings
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            pooling: Pooling strategy (for non-sentence-transformers models)
            show_progress: Show progress bar
        
        Returns:
            numpy array: Matrix of semantic vectors (num_sentences, embedding_dim)
        """
        if self.use_sentence_transformer:
            # Sentence-BERT has built-in batch processing
            with torch.no_grad():
                embeddings = self.model.encode(
                    sentences, 
                    batch_size=batch_size,
                    show_progress_bar=show_progress,
                    convert_to_numpy=True
                )
            return embeddings
        
        # Manual batch processing
        embeddings = []
        num_batches = (len(sentences) + batch_size - 1) // batch_size
        
        iterator = range(0, len(sentences), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding sentences", total=num_batches)
        
        for i in iterator:
            batch = sentences[i:i + batch_size]
            batch_embeddings = [self.encode_sentence(s, max_length, pooling) for s in batch]
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def get_embedding_dim(self):
        """Get the dimensionality of the output embeddings."""
        test_embedding = self.encode_sentence("test")
        return len(test_embedding)


# Example usage functions
def encode_poems(encoder, contents, batch_size=32, save_path=None):
    """
    Encode all poems in the dataset.
    
    Args:
        encoder: SemanticEncoder instance
        contents: List of poem texts
        batch_size: Batch size for processing
        save_path: Path to save embeddings (optional)
    
    Returns:
        numpy array: Semantic embeddings for all poems
    """
    print(f"\nEncoding {len(contents)} poems...")
    embeddings = encoder.encode_batch(contents, batch_size=batch_size)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    if save_path:
        np.save(save_path, embeddings)
        print(f"Saved embeddings to {save_path}")
    
    return embeddings


def compute_similarity(encoder, text1, text2):
    """
    Compute semantic similarity between two texts.
    
    Args:
        encoder: SemanticEncoder instance
        text1: First text
        text2: Second text
    
    Returns:
        float: Cosine similarity score (0 to 1)
    """
    emb1 = encoder.encode_sentence(text1)
    emb2 = encoder.encode_sentence(text2)
    
    # Cosine similarity
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    return similarity

def avg_cosine_similarity(vectors):
    """
    Compute average pairwise cosine similarity among a set of vectors.
    
    Args:
        vectors: numpy array of shape (num_samples, feature_dim)
    
    Returns:
        float: Average cosine similarity
    """
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(vectors)
    # Exclude self-similarity by masking the diagonal
    num_samples = vectors.shape[0]
    sum_sim = np.sum(sim_matrix) - num_samples  # subtract diagonal
    count = num_samples * (num_samples - 1)
    avg_sim = sum_sim / count
    return avg_sim


def k_means(vectors, indices, k, max_iters=1000):
    """
    Perform k-means clustering on the given vectors.
    
    Args:
        vectors: numpy array of shape (num_samples, feature_dim)
        indices: list of cluster indices for each sample
        k: number of clusters
        max_iters: maximum number of iterations

    Returns:
        k lists of indices representing vectors in each cluster
    """
    from sklearn.cluster import KMeans
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=k, max_iter=max_iters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vectors)
    
    # Group indices by cluster
    clusters = [[] for _ in range(k)]
    for idx, label in enumerate(labels):
        clusters[label].append(indices[idx])
    
    return clusters

def find_center_mean(features: np.ndarray) -> np.ndarray:
    """
    Find the center of the features in the feature space.

    Args:
        features: A NumPy array of shape (num_samples, feature_dim).

    Returns:
        A NumPy array representing the center of the features.
    """
    return np.mean(features, axis=0)


import numpy as np

def find_center_geometric_median(features: np.ndarray, eps: float = 1e-5, max_iter: int = 500) -> np.ndarray:
    """
    Find the geometric median of the features in the feature space using Weiszfeld's algorithm.

    Args:
        features: A NumPy array of shape (num_samples, feature_dim).
        eps: Convergence threshold.
        max_iter: Maximum number of iterations.

    Returns:
        A NumPy array representing the geometric median of the features.
    """
    # Initialize with the coordinate-wise median (robust starting point)
    median = np.median(features, axis=0)
    y = median.copy()
    
    for _ in range(max_iter):
        # Compute distances from current estimate
        distances = np.linalg.norm(features - y, axis=1)
        # Avoid division by zero
        nonzero = distances > eps
        if not np.any(nonzero):
            break
        inv_distances = 1.0 / distances[nonzero]
        weights = inv_distances / np.sum(inv_distances)
        y_new = np.sum(weights[:, np.newaxis] * features[nonzero], axis=0)
        
        # Check convergence
        if np.linalg.norm(y - y_new) < eps:
            return y_new
        y = y_new
    
    return y

def calculate_radius_max(features: np.ndarray, center: np.ndarray) -> float:
    """
    Calculate the radius of the features from the center.

    Args:
        features: A NumPy array of shape (num_samples, feature_dim).
        center: A NumPy array representing the center of the features.

    Returns:
        A float representing the radius of the features from the center.
    """
    distances = np.linalg.norm(features - center, axis=1)
    return np.max(distances)

def calculate_radius_two_sigma(features: np.ndarray, center: np.ndarray) -> float:
    """
    Calculate the radius of the features from the center, assuming a Gaussian distribution and using 2 standard deviations.
    The samples inside this radius should cover approximately within 2 sigma of the distribution.
    Args:
        features: A NumPy array of shape (num_samples, feature_dim).
        center: A NumPy array representing the center of the features.

    Returns:
        A float representing the radius of the features from the center.
    """
    distances = np.linalg.norm(features - center, axis=1)
    std_distance = np.std(distances)
    return 2 * std_distance


def get_radius(features: np.ndarray, method: str = 'max', center_method: str = 'mean') -> float:
    """
    Get the radius of the features using specified methods for center and radius calculation.

    Args:
        features: A NumPy array of shape (num_samples, feature_dim).
        method: Method to calculate radius ('max' or 'two_sigma').
        center_method: Method to calculate center ('mean' or 'geometric_median').
    Returns:
        A float representing the radius of the features.
    """
    if center_method == 'mean':
        center = find_center_mean(features)
    elif center_method == 'geometric_median':
        center = find_center_geometric_median(features)
    else:
        raise ValueError(f"Unknown center method: {center_method}")

    if method == 'max':
        radius = calculate_radius_max(features, center)
    elif method == 'two_sigma':
        radius = calculate_radius_two_sigma(features, center)
    else:
        raise ValueError(f"Unknown radius method: {method}")
    return radius

if __name__ == "__main__":
    import pickle
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='sentence-transformers/all-MiniLM-L6-v2', help="Pretrained model name or path")
    parser.add_argument("--dataset_name", type=str, default="minhuh/prh", help="Dataset name to load poems from")
    parser.add_argument("--output", type=str, default="text_semantic_embeddings.npy", help="Path to save output file")
    args = parser.parse_args()

    if args.dataset_name == "SHENJJ1017/poem_aesthetic_eval":
        dataset = load_dataset('SHENJJ1017/poem_aesthetic_eval', split='train', revision='main')
        contents = [entry['content'] for entry in dataset]
        data_type = 'poem'
    else:
        dataset = load_dataset('minhuh/prh', split='train', revision='wit_1024')
        contents = [str(x['text'][0]) for x in dataset]
        data_type = 'text'
    
    words = sum([x.split() for x in contents], [])
    print(f"Loaded {len(contents)} texts with a total of {len(words)} words.")


    encoder = SemanticEncoder(model_name=args.model_name)
    vectors = encode_poems(encoder, contents, batch_size=64, save_path=args.output)
    vectors = np.load(args.output)
    k = 2
    clusters = k_means(vectors, list(range(len(vectors))), k=k)

    with open(f"figs/clusters/{data_type}_clusters_{k}.pkl", "wb") as f:
        pickle.dump(clusters, f)
    for i, cluster in enumerate(clusters):
        selected = [contents[idx] for idx in cluster]
        words = sum([content.split() for content in selected], [])
        word_cloud(words, file_path=f"figs/clusters/{data_type}_wordcloud_cluster_{i}_in_{k}_.png")

    k = 2
    clusters = pickle.load(open(f"figs/clusters/{data_type}_clusters_{k}.pkl", "rb"))
    vectors = np.load(args.output)

    all_avg_sim = avg_cosine_similarity(vectors)
    print(f"Overall average cosine similarity: {all_avg_sim:.4f}")
    radius = get_radius(np.array(vectors), method='two_sigma', center_method='geometric_median')
    print(f"Overall radius (two_sigma, geometric_median): {radius:.4f}")
    for i, cluster in enumerate(clusters):
        cluster_vectors = vectors[cluster]
        cluster_avg_sim = avg_cosine_similarity(cluster_vectors)
        print(f"Cluster {i} average cosine similarity: {cluster_avg_sim:.4f}")
        radius = get_radius(np.array(cluster_vectors), method='two_sigma', center_method='geometric_median')
        print(f"Cluster {i} radius (two_sigma, geometric_median): {radius:.4f}")