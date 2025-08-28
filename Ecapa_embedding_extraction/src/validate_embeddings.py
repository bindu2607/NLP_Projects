#!/usr/bin/env python3
import numpy as np
import os
import glob
from sklearn.metrics import roc_curve
from scipy.spatial.distance import cosine
import itertools
import argparse
import logging
from pathlib import Path

DEFAULT_IN_DIR = Path(r"D:\Oscowl ai\AISHELL3 Speaker Embedding Extractor day8\processed\embedding_extraction\test\embeddings")

def load_embeddings(root_dir, emb_dim=192, emb_dtype=np.float32, logger=None):
    """
    Load embeddings per speaker as a list.
    Returns dict: data[speaker] = list of embeddings
    """
    data = {}
    for spk_dir in os.listdir(root_dir):
        spk_path = os.path.join(root_dir, spk_dir)
        if not os.path.isdir(spk_path):
            continue
        embeddings = []
        for emb_file in glob.glob(os.path.join(spk_path, "*.npy")):
            emb = np.load(emb_file)
            if emb.shape != (emb_dim,):
                if logger:
                    logger.warning(f"Skipping {emb_file} due to unexpected shape {emb.shape}")
                continue
            if emb.dtype != emb_dtype:
                emb = emb.astype(emb_dtype)

            # Normalize embedding
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm

            embeddings.append(emb)
        if embeddings:
            data[spk_dir] = embeddings
    return data

def cosine_sim(a, b):
    """
    Compute cosine similarity between two normalized vectors.
    """
    return 1 - cosine(a, b)

def compute_pairs(data):
    """
    Compute all same-speaker pairs and different-speaker pairs.
    Warning: can be very large for many embeddings!
    """
    same_pairs = []
    diff_pairs = []

    speakers = list(data.keys())
    for spk in speakers:
        emb_list = data[spk]
        for i, j in itertools.combinations(range(len(emb_list)), 2):
            same_pairs.append((emb_list[i], emb_list[j]))

    for spk1, spk2 in itertools.combinations(speakers, 2):
        emb1 = data[spk1]
        emb2 = data[spk2]
        for e1 in emb1:
            for e2 in emb2:
                diff_pairs.append((e1, e2))

    return same_pairs, diff_pairs

def calc_scores(pairs):
    """
    Calculate cosine similarity scores for all pairs.
    """
    return [cosine_sim(a, b) for a, b in pairs]

def compute_eer(scores, labels):
    """
    Compute Equal Error Rate (EER) given scores and true labels.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_threshold_idx = (np.abs(fnr - fpr)).argmin()
    eer = max(fpr[eer_threshold_idx], fnr[eer_threshold_idx])
    return eer

def main(args):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("EvalScript")

    emb_root = args.emb_root if args.emb_root else DEFAULT_IN_DIR
    logger.info(f"Loading embeddings from {emb_root}")
    data = load_embeddings(emb_root, emb_dim=args.emb_dim, emb_dtype=np.float32, logger=logger)
    if not data:
        logger.error("No embeddings loaded. Exiting.")
        return

    logger.info(f"Loaded embeddings for {len(data)} speakers")

    logger.info("Computing pairs...")
    same_pairs, diff_pairs = compute_pairs(data)
    logger.info(f"Same-speaker pairs: {len(same_pairs)}")
    logger.info(f"Different-speaker pairs: {len(diff_pairs)}")

    logger.info("Calculating cosine similarity scores...")
    same_scores = calc_scores(same_pairs)
    diff_scores = calc_scores(diff_pairs)

    scores = np.array(same_scores + diff_scores)
    labels = np.array([1]*len(same_scores) + [0]*len(diff_scores))

    logger.info("Computing EER...")
    eer = compute_eer(scores, labels)

    mean_intra = np.mean(same_scores)
    mean_inter = np.mean(diff_scores)

    logger.info(f"Overall EER: {eer:.4f}")
    logger.info(f"Mean intra-speaker cosine similarity: {mean_intra:.4f}")
    logger.info(f"Mean inter-speaker cosine similarity: {mean_inter:.4f}")

    # Pretty print results in a box
    print("\n" + "="*40)
    print("      Speaker Embedding Evaluation      ")
    print("="*40)
    print(f"Mean Intra-speaker Similarity : {mean_intra:.4f}")
    print(f"Mean Inter-speaker Similarity : {mean_inter:.4f}")
    print(f"Overall Equal Error Rate (EER): {eer:.4f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate speaker embeddings: cosine similarity and EER")
    parser.add_argument("--emb-root", type=str, required=False,
                        help=f"Root directory with speaker subfolders containing .npy embeddings (default: {DEFAULT_IN_DIR})")
    parser.add_argument("--emb-dim", type=int, default=192, help="Expected embedding dimension")
    args = parser.parse_args()

    main(args)
