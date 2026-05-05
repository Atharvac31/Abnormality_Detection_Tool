import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# 1. Entropy Function
# -----------------------------
def compute_entropy(probs):
    """
    probs: numpy array (N,) → probability of class 1
    """
    probs = np.clip(probs, 1e-9, 1 - 1e-9)
    return -(probs * np.log(probs) + (1 - probs) * np.log(1 - probs))


# -----------------------------
# 2. Entropy Filtering
# -----------------------------
def entropy_filtering(features, probs, filter_ratio=0.2):
    """
    Keep top uncertain samples
    """
    entropy_vals = compute_entropy(probs)

    k = int(len(features) * filter_ratio)
    indices = np.argsort(entropy_vals)[-k:]

    return features[indices], probs[indices], indices


# -----------------------------
# 3. Greedy Facility Location
# -----------------------------
def greedy_facility_location(features, budget):
    """
    Select diverse samples using greedy facility location
    """
    sim = cosine_similarity(features)
    N = len(features)

    selected = []
    current_max = np.zeros(N)

    for _ in range(min(budget, N)):
        best_gain = -1
        best_idx = None

        for j in range(N):
            if j in selected:
                continue

            gain = np.sum(np.maximum(current_max, sim[:, j]) - current_max)

            if gain > best_gain:
                best_gain = gain
                best_idx = j

        selected.append(best_idx)
        current_max = np.maximum(current_max, sim[:, best_idx])

    return selected


# -----------------------------
# 4. Main SPEAR-SS Function
# -----------------------------
def spear_ss_selection(
    features,
    probs,
    budget=200,
    filter_ratio=0.2,
    mode="supervised"
):
    """
    features: (N, D)
    probs: (N,) probability of abnormal
    budget: number of samples to select
    filter_ratio: % of uncertain samples to keep
    mode: 'supervised' or 'unsupervised'
    
    returns: selected_indices (original indices)
    """

    # Step 1: Entropy filtering
    f_features, f_probs, f_indices = entropy_filtering(
        features, probs, filter_ratio
    )

    if len(f_features) == 0:
        print("⚠️ No samples after filtering")
        return []

    # Step 2: Selection
    if mode == "unsupervised":
        selected_local = greedy_facility_location(f_features, budget)

    elif mode == "supervised":
        # Split into classes using pseudo labels
        pseudo_labels = (f_probs > 0.5).astype(int)

        idx_0 = np.where(pseudo_labels == 0)[0]
        idx_1 = np.where(pseudo_labels == 1)[0]

        half = budget // 2

        selected_local = []

        if len(idx_0) > 0:
            sel_0 = greedy_facility_location(f_features[idx_0], min(half, len(idx_0)))
            selected_local.extend(idx_0[sel_0])

        if len(idx_1) > 0:
            sel_1 = greedy_facility_location(f_features[idx_1], min(half, len(idx_1)))
            selected_local.extend(idx_1[sel_1])

    else:
        raise ValueError("mode must be 'supervised' or 'unsupervised'")

    # Map back to original indices
    selected_indices = f_indices[selected_local]

    return selected_indices


# -----------------------------
# 5. Debug Helper
# -----------------------------
def debug_selection(features, probs, selected_indices):
    """
    Print useful info about selected samples
    """
    print("Total samples:", len(features))
    print("Selected samples:", len(selected_indices))

    selected_probs = probs[selected_indices]

    print("Avg probability:", np.mean(selected_probs))
    print("Min probability:", np.min(selected_probs))
    print("Max probability:", np.max(selected_probs))

    print("Example selected indices:", selected_indices[:10])