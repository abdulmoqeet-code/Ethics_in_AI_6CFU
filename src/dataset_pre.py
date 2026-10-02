import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from src.pre_eval import evaluate_cv_tier

# ==========================================
# 1. CATEGORIZATION & DATASET SAVING
# ==========================================

def categorize_and_save_cvs(cv_list, model, tokenizer, output_json_path):
    """Processes CV list through Qwen tier evaluation, appends the assigned tier

    (Confident, Average, Weak), and saves the updated dataset to JSON.
    """
    print(f"\n--- Stratifying {len(cv_list)} CVs into Qualification Tiers ---")
    updated_cv_list = []

    for idx, cv_item in enumerate(cv_list, start=1):
        resume_id = cv_item["resume_id"]
        cv_text = cv_item["cv_text"]

        # Run Qwen tier evaluation (returns string: "Confident", "Average", or "Weak")
        tier = evaluate_cv_tier(cv_text, model, tokenizer)

        # Append tier field directly to existing item dictionary
        updated_item = dict(cv_item)
        updated_item["tier"] = tier
        updated_cv_list.append(updated_item)

        print(f"[{idx}/{len(cv_list)}] Resume ID: {resume_id} | Tier: {tier}")

    # Save categorized dataset to file
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(updated_cv_list, f, indent=4, ensure_ascii=False)

    print(f"\nCategorized dataset saved successfully to: '{output_json_path}'")
    return updated_cv_list


# ==========================================
# 2. K-MEANS DIVERSITY SELECTION
# ==========================================



import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


def create_tiered_distance_groups(
    cv_list,
    n_groups=5,
    clusters_per_tier=12,
    embedding_model_name="all-MiniLM-L6-v2",
    output_dir="./distance_ranked_groups",
):
    """1. Groups input CVs by tier (Confident, Average, Weak).

    2. Fits exactly `clusters_per_tier` (12) K-Means clusters on ALL CVs in each tier.
    3. Ranks CVs within each cluster by Euclidean distance to its centroid.
    4. Distributes rank 0 to Group 1, rank 1 to Group 2, ..., rank N-1 to Group N.
       (Each group receives 12 CVs per tier = 36 CVs per group).
    5. Saves each group to JSON and prints tier breakdown summaries.
    """
    os.makedirs(output_dir, exist_ok=True)
    embedder = SentenceTransformer(embedding_model_name)

    # Step 1: Separate CVs into tier pools
    tiered_pools = {}
    for cv in cv_list:
        tier = cv.get("tier", "Uncategorized")
        tiered_pools.setdefault(tier, []).append(cv)

    print(f"\n--- Fitting {clusters_per_tier} Clusters per Tier across {len(cv_list)} Total CVs ---")

    # Initialize N groups
    groups = [[] for _ in range(n_groups)]

    # Step 2: Fit 12 Clusters per Tier and Partition into N Distance Groups
    for tier, pool in tiered_pools.items():
        print(f"\nProcessing Tier: '{tier}' ({len(pool)} total CVs)")

        if len(pool) < clusters_per_tier:
            raise ValueError(
                f"Tier '{tier}' has only {len(pool)} CVs, but {clusters_per_tier} clusters were requested."
            )

        corpus = [item["cv_text"] for item in pool]
        embeddings = embedder.encode(corpus, show_progress_bar=False)

        # Fit exactly 12 clusters for this tier
        kmeans = KMeans(n_clusters=clusters_per_tier, random_state=42, n_init=10)
        kmeans.fit(embeddings)

        cluster_labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        # Calculate Euclidean distance from each CV to its assigned cluster center
        distances = np.linalg.norm(embeddings - centers[cluster_labels], axis=1)

        # Collect and sort CVs within each of the 12 clusters by distance
        cluster_buckets = {c_id: [] for c_id in range(clusters_per_tier)}
        for idx, (c_id, dist) in enumerate(zip(cluster_labels, distances)):
            cluster_buckets[c_id].append((dist, pool[idx]))

        for c_id in cluster_buckets:
            cluster_buckets[c_id].sort(key=lambda x: x[0])  # Sort closest to furthest

        # Assign Rank 0 -> Group 1, Rank 1 -> Group 2, ..., Rank N-1 -> Group N
        for c_id, sorted_items in cluster_buckets.items():
            for group_idx in range(n_groups):
                if group_idx < len(sorted_items):
                    _, cv_item = sorted_items[group_idx]
                    groups[group_idx].append(cv_item)

    # Step 3: Print Summaries & Export Group JSON Files
    print(f"\n================ SELECTION SUMMARY ================")
    saved_filepaths = []

    for g_idx, group_cvs in enumerate(groups, start=1):
        tier_counts = {}
        resume_ids = []
        for cv in group_cvs:
            t = cv.get("tier", "Uncategorized")
            tier_counts[t] = tier_counts.get(t, 0) + 1
            resume_ids.append(cv["resume_id"])

        rank_label = "Most Central (#1 Medoids)" if g_idx == 1 else f"#{g_idx} Distance Rank"
        print(f"\n📁 GROUP {g_idx} [{rank_label}]")
        print(f"   Total Selected CVs : {len(group_cvs)}")
        print(f"   Tier Breakdown     : {tier_counts}")
        print(f"   Resume IDs ({len(resume_ids)}) : {resume_ids}")

        # Save Group JSON
        file_name = f"group_{g_idx}_tier_selection.json"
        file_path = os.path.join(output_dir, file_name)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(group_cvs, f, indent=4, ensure_ascii=False)

        saved_filepaths.append(file_path)
        print(f"   Saved File         : '{file_path}'")

    print(f"\nSuccessfully created and saved {n_groups} group files in '{output_dir}'.")
    return groups