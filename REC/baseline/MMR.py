import numpy as np
from tqdm import tqdm

class MMR(object):
    def __init__(self, test_num):
        """
        :param test_num: upper bound of samples to process during testing
        """
        self.max_len = 100
        self.test_num = test_num

    def softmax(self, x: np.array):
        
        if x.ndim == 1:
            x = x.reshape(1, -1)
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def _precompute_jaccard_matrix(self, multi_hot_vectors):
        """
        compute Jaccard similarity matrix for given multi-hot vectors
        :param multi_hot_vectors: numpy array of shape (N, D) where N is number of items and D is dimension of multi-hot vector
        :return: similarity matrix of shape (N, N)
        """
      
        B = (multi_hot_vectors > 0).astype(float)
        intersection = B @ B.T
        row_sums = np.sum(B, axis=1)
        union = row_sums[:, None] + row_sums[None, :] - intersection
        with np.errstate(divide='ignore', invalid='ignore'):
            similarity_matrix = np.divide(intersection, union) 
            similarity_matrix = np.nan_to_num(similarity_matrix) 
            
        return similarity_matrix

    def post_process(self, output_dict, id_multihot, lamda=1.0):
        """
        MMR post-processing to enhance diversity in recommendation results
        :param output_dict: model output dictionary containing 'predictions' and 'dataset_ids'
        :param id_multihot: dictionary mapping item IDs to their multi-hot feature vectors
        :param lamda: trade-off parameter between relevance and diversity`
        """
        rank_scores_raw = output_dict["predictions"]
        batch_ids = output_dict["dataset_ids"]
        batchsize = len(rank_scores_raw)

        process_size = min(int(self.test_num), batchsize)
        
        sample_rank_scores = rank_scores_raw[:process_size]
        sample_batch_ids = batch_ids[:process_size]
        
        norm_scores_all = self.softmax(sample_rank_scores)
        final_predictions = np.zeros((process_size, sample_rank_scores.shape[1]))

        print(f"Running MMR post-processing (lamda={lamda})...")
        for i in tqdm(range(process_size)):
            scores = norm_scores_all[i]
            ids = sample_batch_ids[i]
            seq_len = len(ids)
            try:
                vectors = np.array([id_multihot.get(str(int(x)), np.zeros(1)) for x in ids])
            except:
                vectors = np.array([id_multihot.get(str(x), np.zeros(1)) for x in ids])
                
            sim_matrix = self._precompute_jaccard_matrix(vectors)
            chosen_indices = []
            remaining_indices = list(range(min(seq_len, self.max_len)))
            while len(remaining_indices) > 0:
                if not chosen_indices:
                    next_idx = remaining_indices[np.argmax(scores[remaining_indices])]
                else:
                    relevance = scores[remaining_indices]
                    candidate_sims = sim_matrix[remaining_indices][:, chosen_indices]
                    max_sim = np.max(candidate_sims, axis=1)
                    mmr_vals = lamda * relevance - (1 - lamda) * max_sim
                    next_idx = remaining_indices[np.argmax(mmr_vals)]
                
                chosen_indices.append(next_idx)
                remaining_indices.remove(next_idx)
            for rank, original_idx in enumerate(chosen_indices):
                final_predictions[i, original_idx] = 1.0 - (rank / len(chosen_indices))
        return {
            "predictions": final_predictions,
            "dataset_ids": sample_batch_ids  
        }
    