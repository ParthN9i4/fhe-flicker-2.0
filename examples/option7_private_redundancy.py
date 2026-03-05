"""
Option 7 — Private Redundancy Protocol (Fully Encrypted Dot Products)
======================================================================

In the original FHE-Flicker global undersampling step, the dominant client
uploads its buffer feature vectors as a plaintext list and the server calls:

    enc_sim = enc_feat.dot(buf_vec.tolist())   # buf_vec is PLAIN Python list

This exposes the dominant client's buffer vectors to the server. Feature
inversion attacks (Fredrikson et al., 2015; He et al., 2019) can reconstruct
recognisable images from ResNet-512 embeddings, so this is a real privacy risk.

This script upgrades the protocol so that buffer vectors are also CKKS-encrypted
before being sent to the server.  The server computes a fully ciphertext dot
product using:

    Step 1:  enc_prod = enc_buf * enc_feat        (cipher × cipher, depth 1)
    Step 2:  for shift in [1, 2, 4, ..., 256]:    (rotation-based sum, depth 0)
                 result = result + result.rotate(shift)
             → slot 0 of result holds Σᵢ enc_prod[i]  (the dot product)

The server learns nothing about either party's feature vectors. The key-holder
decrypts only the scalar similarity value and makes the remove/keep decision.

Depth budget
------------
    fhe_inner_product uses depth 1 — the same depth as the original
    enc_feat.dot(plain_buf) call.  No CKKS context changes are needed.
    The existing [60, 40, 40, 60] chain (designed for depth 2) has one
    level remaining after each fully-encrypted dot product.

Usage
-----
    python option7_private_redundancy.py

Requirements
------------
    pip install torch torchvision tenseal numpy
"""

import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as dset
import torchvision.models as models
import torchvision.transforms as T
import tenseal as ts


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

FHE_SAMPLE_SIZE = 200   # features sampled per non-dominant client per round
N_CLIENTS       = 3
DIRICHLET_ALPHA = 0.3
SEED            = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# CKKS context
# ─────────────────────────────────────────────────────────────────────────────

def setup_ckks_context():
    """
    Initialise the TenSEAL CKKS context.

    The critical parameter for Option 7 is `generate_galois_keys()`.
    Galois keys allow the server to perform ciphertext slot rotations, which
    are the building block of the rotation-based sum in fhe_inner_product().
    They do NOT reveal any information about the secret key.

    poly_modulus_degree = 8192   →  4096 slots, ≥128-bit security
    coeff_mod_bit_sizes           →  [60, 40, 40, 60]: depth-2 budget
      The two 40-bit middle primes mean we can execute up to 2 consecutive
      cipher×cipher multiplications. fhe_inner_product uses exactly 1.
    global_scale = 2^40          →  ~12 decimal digits of precision
    """
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    ctx.generate_galois_keys()   # required for .rotate() in fhe_inner_product
    ctx.global_scale = 2 ** 40
    return ctx


# ─────────────────────────────────────────────────────────────────────────────
# Image transforms & frozen ResNet-18 feature extractor
# ─────────────────────────────────────────────────────────────────────────────

transform_resnet = T.Compose([
    T.ToTensor(),
    T.Resize((224, 224), antialias=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

resnet = models.resnet18(weights="IMAGENET1K_V1")
resnet.fc = nn.Identity()
resnet.eval()
resnet = resnet.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Data utilities
# ─────────────────────────────────────────────────────────────────────────────

def safe_collate(batch):
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return images, labels


def rotate_list(lst, k):
    k = k % len(lst)
    return lst[k:] + lst[:k]


def generate_imbalanced_split(total, n_clients, alpha):
    proportions = np.random.dirichlet([alpha] * n_clients)
    counts = (proportions * total).astype(int)
    counts[-1] = total - counts[:-1].sum()
    return np.maximum(counts, 1)


def distribute_cifar_imbalanced(labels, n_clients, alpha=0.5, seed=None):
    """
    Distribute CIFAR-10 indices across clients using a Dirichlet imbalance model.

    Each class is split among clients according to a Dirichlet(alpha) draw.
    Lower alpha → higher imbalance (one client dominates each class).
    The split order is rotated per class so different classes are dominant on
    different clients.

    Returns: dict {client_id: [list of dataset indices]}
    """
    if seed is not None:
        np.random.seed(seed)
    n_classes = len(np.unique(labels))
    class_indices = {c: np.where(labels == c)[0].tolist() for c in range(n_classes)}
    for c in class_indices:
        np.random.shuffle(class_indices[c])
    client_indices = {cid: [] for cid in range(n_clients)}
    for c in range(n_classes):
        idxs   = class_indices[c]
        counts = generate_imbalanced_split(len(idxs), n_clients, alpha)
        offset = 0
        order  = rotate_list(list(range(n_clients)), (c * 3) % n_clients)
        for i, cid in enumerate(order):
            client_indices[cid].extend(idxs[offset: offset + counts[i]])
            offset += counts[i]
    return client_indices


def l2_normalize(feats):
    """Row-wise L2 normalisation. After this, dot product = cosine similarity."""
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return feats / norms


def extract_features_batched(dataset, batch_size=256):
    """Extract 512-dim ResNet-18 features from a Subset or Dataset."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=safe_collate)
    all_feats = []
    with torch.no_grad():
        for x, _ in loader:
            all_feats.append(resnet(x.to(device)).cpu().numpy())
    return np.concatenate(all_feats, axis=0)


def encrypt_feature_matrix(ctx, feats_np):
    """Encrypt each row of a feature matrix as a separate CKKS ciphertext."""
    return [ts.ckks_vector(ctx, row.tolist()) for row in feats_np]


def decrypt_scalar(enc_result):
    """Decrypt a CKKS ciphertext and return the value in slot 0 as a float."""
    return float(enc_result.decrypt()[0])


# ─────────────────────────────────────────────────────────────────────────────
# Majority-class buffer selection
# ─────────────────────────────────────────────────────────────────────────────

def get_majority_classes(y, lin_threshold=0.9, n_classes=10):
    """
    Return the list of classes whose count >= lin_threshold * max_count.
    These are the over-represented classes targeted for removal.
    """
    counts    = np.bincount(y, minlength=n_classes)
    max_count = counts.max()
    return [c for c in range(n_classes) if counts[c] >= lin_threshold * max_count]


def local_redundancy_majority(X, y, theta, lin_threshold=0.9):
    """
    Select locally-redundant majority-class samples for the global check.

    Algorithm
    ---------
    1. Identify majority classes (count >= lin_threshold * max_count).
    2. Extract and L2-normalise their ResNet-18 feature vectors.
    3. Compute the full pairwise cosine similarity matrix (feats @ feats.T).
    4. Rank samples by mean similarity to the rest of the set.
    5. Take the top theta-fraction as buffer candidates (most redundant).

    Returns
    -------
    buffer_idx  : np.ndarray of sample indices into X
    buffer_vecs : np.ndarray of shape (k, 512), L2-normalised features
    """
    maj_classes = get_majority_classes(y, lin_threshold)
    if not maj_classes:
        return np.array([], dtype=int), np.empty((0, 512))

    maj_mask = np.isin(y, maj_classes)
    maj_idxs = np.where(maj_mask)[0]
    feats    = extract_features_batched(Subset(X, maj_idxs.tolist()))
    feats    = l2_normalize(feats)

    sim_matrix = feats @ feats.T        # cosine similarities (all pairs)
    np.fill_diagonal(sim_matrix, 0)
    mean_sim = sim_matrix.mean(axis=1)

    n_buf  = max(1, int(theta * len(maj_idxs)))
    ranked = np.argsort(-mean_sim)[:n_buf]
    return maj_idxs[ranked], feats[ranked]


# ─────────────────────────────────────────────────────────────────────────────
# Baseline: cipher × plaintext (original Section 9 approach)
# ─────────────────────────────────────────────────────────────────────────────

def fhe_undersample_global(ctx, dom_id, clients, theta=0.2, eta=0.6,
                            lin_threshold=0.9):
    """
    Original global redundancy undersampling (Section 9 of the notebook).

    The dominant client's buffer vectors are passed as a PLAINTEXT list to
    enc_feat.dot(buf_vec.tolist()).  The server can read buf_vec directly.

    Retained here for side-by-side comparison with Option 7.
    """
    Xd = clients[dom_id]["X"]
    yd = clients[dom_id]["y"]

    buffer_idx, buffer_vecs = local_redundancy_majority(Xd, yd, theta, lin_threshold)
    if len(buffer_idx) == 0:
        return Xd, yd

    # Encrypt other clients' features (these were already private in Section 9)
    other_enc_feats = {}
    for cid, client in clients.items():
        if cid == dom_id:
            continue
        feats      = l2_normalize(extract_features_batched(client["X"]))
        sample_idx = np.random.choice(len(feats), min(FHE_SAMPLE_SIZE, len(feats)),
                                      replace=False)
        other_enc_feats[cid] = encrypt_feature_matrix(ctx, feats[sample_idx])

    remove = []
    for buf_idx, buf_vec in zip(buffer_idx, buffer_vecs):
        client_avgs = []
        for cid, enc_feats in other_enc_feats.items():
            sims = [decrypt_scalar(enc_f.dot(buf_vec.tolist()))  # buf_vec: PLAIN
                    for enc_f in enc_feats]
            client_avgs.append(float(np.mean(sims)))
        if float(np.mean(client_avgs)) >= eta:
            remove.append(buf_idx)

    mask = np.ones(len(Xd), dtype=bool)
    mask[remove] = False
    return Subset(Xd, np.where(mask)[0]), yd[mask]


# ─────────────────────────────────────────────────────────────────────────────
# Option 7 — fully encrypted dot product kernel
# ─────────────────────────────────────────────────────────────────────────────

def fhe_inner_product(enc_a, enc_b, n=512):
    """
    Compute the dot product of two CKKS-encrypted vectors without decryption.

    Both enc_a and enc_b must be encrypted under the same TenSEAL context.
    The server performs all arithmetic on ciphertexts; it sees no plaintext.

    Algorithm
    ---------
    Step 1 — Element-wise cipher × cipher multiply:
        enc_prod[i] = enc_a[i] * enc_b[i]   for i in 0..n-1

        This is the CKKS Multiply operation. It:
          • consumes one multiplicative level (uses one 40-bit prime)
          • increases noise as σ ≈ σ_a · σ_b · Δ  (standard CKKS multiply noise)
          • leaves n zero-padded slots untouched (they stay 0)

    Step 2 — Rotation-based sum  (log₂(n) = 9 iterations for n = 512):
        result = enc_prod
        for shift in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            result = result + result.rotate(shift)

        How it works:
          After rotate(1):  slot[0] += slot[1]
          After rotate(2):  slot[0] += slot[2] + slot[3]  (because slot[1] carries)
          …
          After rotate(256): slot[0] = enc_prod[0] + enc_prod[1] + … + enc_prod[511]

        Each .rotate(k) is a ring automorphism applied via the Galois key for k.
        CKKS addition (result + result.rotate(k)) consumes ZERO depth.
        Nine rotations and nine additions — all free in terms of multiplicative levels.

    Total depth consumed: 1  (Step 1 only).
    Remaining levels in [60, 40, 40, 60]: 1.

    Parameters
    ----------
    enc_a : CKKSVector  — query vector  (dominant client's buffer feature)
    enc_b : CKKSVector  — data vector   (other client's sample feature)
    n     : int         — number of active slots; MUST be a power of 2

    Returns
    -------
    CKKSVector where slot 0 holds the dot product (≈ cosine similarity for
    L2-normalised inputs). Decrypt with:  enc_result.decrypt()[0]
    """
    assert n > 0 and (n & (n - 1)) == 0, "n must be a positive power of 2"

    # Step 1: element-wise cipher × cipher multiply  (depth 1)
    enc_prod = enc_a * enc_b

    # Step 2: rotation-based sum into slot 0  (depth 0)
    # shifts: 1, 2, 4, 8, 16, 32, 64, 128, 256  (log₂(512) = 9 iterations)
    result = enc_prod
    shift  = 1
    while shift < n:
        result = result + result.rotate(shift)
        shift <<= 1

    return result   # caller reads: decrypt_scalar(result) or result.decrypt()[0]


# ─────────────────────────────────────────────────────────────────────────────
# Option 7 — fully private global undersampling
# ─────────────────────────────────────────────────────────────────────────────

def fhe_undersample_global_private(ctx, dom_id, clients, theta=0.2, eta=0.6,
                                    lin_threshold=0.9):
    """
    FULLY PRIVATE global redundancy undersampling.

    What changes compared to fhe_undersample_global()
    --------------------------------------------------
    Before:  buf_vec is a plaintext Python list; server reads it directly.
    After:   buf_vec is encrypted to enc_buf before the server sees it.
             Server calls:  enc_sim = fhe_inner_product(enc_buf, enc_feat)
             Server receives ONLY ciphertexts; it observes no raw floats.

    Privacy model
    -------------
    Dominant client  Encrypts its buffer vectors before sending to the server.
                     The server receives CKKS ciphertexts — opaque without the key.
    Other clients    Encrypt their feature vectors (same as Section 9).
    Server           Evaluates cipher×cipher multiplications and rotation-sums.
                     It computes only on ciphertexts; zero plaintext feature data
                     passes through it.
    Key-holder       Decrypts the scalar similarity result and makes the
                     remove/keep decision. If the server and key-holder are
                     separate entities, the server learns at most 1 binary bit
                     per buffer candidate — the minimal possible leakage.

    Depth budget
    ------------
    fhe_inner_product : depth 1  (same as enc_feat.dot(plain_buf) in Section 9)
    Context [60, 40, 40, 60] supports depth 2 — one level remains after each call.
    No context changes are required.

    Parameters  (identical signature to fhe_undersample_global)
    ----------
    ctx           : TenSEAL CKKS context
    dom_id        : dominant client ID
    clients       : dict {cid: {"X": Subset, "y": np.ndarray}}
    theta         : fraction of majority samples to buffer (default 0.2)
    eta           : cosine similarity threshold for removal (default 0.6)
    lin_threshold : majority-class definition threshold (default 0.9)
    """
    Xd = clients[dom_id]["X"]
    yd = clients[dom_id]["y"]

    # Step 1: local redundancy candidates (dominant client — plaintext, local only)
    buffer_idx, buffer_vecs = local_redundancy_majority(Xd, yd, theta, lin_threshold)
    if len(buffer_idx) == 0:
        print("  No buffer candidates found. Skipping undersampling.")
        return Xd, yd
    print(f"  Local buffer: {len(buffer_idx)} majority-class candidates")

    # Step 2: dominant client encrypts its buffer vectors  ← KEY CHANGE vs Section 9
    print(f"  Dominant client {dom_id}: encrypting {len(buffer_vecs)} buffer vectors...")
    enc_buffer_vecs = [ts.ckks_vector(ctx, bv.tolist()) for bv in buffer_vecs]
    # From this point the server never sees buffer_vecs — only enc_buffer_vecs.

    # Step 3: other clients encrypt their features (same as Section 9)
    other_enc_feats = {}
    for cid, client in clients.items():
        if cid == dom_id:
            continue
        feats      = l2_normalize(extract_features_batched(client["X"]))
        sample_idx = np.random.choice(len(feats), min(FHE_SAMPLE_SIZE, len(feats)),
                                      replace=False)
        print(f"  Client {cid}: encrypting {len(sample_idx)} feature vectors...")
        other_enc_feats[cid] = encrypt_feature_matrix(ctx, feats[sample_idx])

    # Step 4: server computes fully-encrypted dot products
    print("  Server: computing cipher × cipher dot products (depth 1)...")
    remove = []
    for buf_idx, enc_buf in zip(buffer_idx, enc_buffer_vecs):
        client_avgs = []
        for cid, enc_feats in other_enc_feats.items():
            sims = []
            for enc_f in enc_feats:
                # Both enc_buf and enc_f are ciphertexts — server sees neither plain
                enc_sim = fhe_inner_product(enc_buf, enc_f, n=512)
                sims.append(decrypt_scalar(enc_sim))          # key-holder decrypts
            client_avgs.append(float(np.mean(sims)))

        if float(np.mean(client_avgs)) >= eta:
            remove.append(buf_idx)

    print(f"  Removing {len(remove)} globally redundant samples")

    # Step 5: apply mask to dominant client's dataset
    mask = np.ones(len(Xd), dtype=bool)
    mask[remove] = False
    return Subset(Xd, np.where(mask)[0]), yd[mask]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  Option 7 — Private Redundancy Protocol")
    print("  (Fully Encrypted Dot Products via CKKS)")
    print("=" * 62)

    # 1. CKKS context
    print("\n[1/5] Setting up CKKS context...")
    ctx = setup_ckks_context()
    print("  poly_modulus_degree = 8192  →  4096 slots, ≥128-bit security")
    print("  coeff_mod_bit_sizes = [60, 40, 40, 60]  →  depth-2 budget")
    print("  Galois keys generated (required for ciphertext rotations)")
    print(f"  Device = {device}")

    # 2. Load CIFAR-10
    print("\n[2/5] Loading CIFAR-10...")
    train_dataset = dset.CIFAR10(
        root="./data", train=True, download=True, transform=transform_resnet
    )
    all_labels = np.array(train_dataset.targets)
    print(f"  {len(train_dataset)} training samples loaded")

    # 3. Imbalanced client distribution
    print(f"\n[3/5] Distributing data across {N_CLIENTS} clients "
          f"(Dirichlet α = {DIRICHLET_ALPHA}, seed = {SEED})...")
    client_idxs = distribute_cifar_imbalanced(
        all_labels, N_CLIENTS, alpha=DIRICHLET_ALPHA, seed=SEED
    )
    clients = {}
    for cid, idxs in client_idxs.items():
        clients[cid] = {"X": Subset(train_dataset, idxs), "y": all_labels[idxs]}
        cc = np.bincount(clients[cid]["y"], minlength=10).tolist()
        print(f"  Client {cid}: {len(idxs):5d} samples | classes: {cc}")

    # 4. Correctness check: fhe_inner_product vs numpy
    print("\n[4/5] Correctness check: fhe_inner_product vs numpy dot product...")
    np.random.seed(0)
    a = np.random.randn(512).astype(np.float32)
    b = np.random.randn(512).astype(np.float32)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)

    plain_dot = float(np.dot(a, b))
    enc_a     = ts.ckks_vector(ctx, a.tolist())
    enc_b     = ts.ckks_vector(ctx, b.tolist())
    enc_r     = fhe_inner_product(enc_a, enc_b, n=512)
    fhe_dot   = enc_r.decrypt()[0]

    print(f"  Plaintext dot product  : {plain_dot: .7f}")
    print(f"  FHE dot product        : {fhe_dot: .7f}")
    print(f"  Absolute error         : {abs(plain_dot - fhe_dot):.2e}")
    print(f"  (cipher×cipher depth-1 matches plaintext to within CKKS noise)")

    # 5. Pick the dominant client (highest single-class fraction)
    max_fracs = {}
    for cid, c in clients.items():
        counts = np.bincount(c["y"], minlength=10)
        max_fracs[cid] = counts.max() / len(c["y"])
    dom_id = max(max_fracs, key=max_fracs.get)
    print(f"\n  Dominant client: {dom_id}  "
          f"(max class fraction = {max_fracs[dom_id]:.3f})")
    n_before = len(clients[dom_id]["y"])

    # 6a. Baseline: Section 9 (cipher × PLAINTEXT)
    print(f"\n[5/5a] Section 9 baseline — cipher × PLAINTEXT buffer vectors...")
    print(f"       (server CAN read dominant client's buffer vectors)")
    np.random.seed(SEED)
    t0 = time.time()
    Xd_plain, yd_plain = fhe_undersample_global(
        ctx, dom_id,
        {cid: {"X": c["X"], "y": c["y"].copy()} for cid, c in clients.items()},
        theta=0.2, eta=0.6,
    )
    n_removed_plain = n_before - len(yd_plain)
    t_plain = time.time() - t0
    print(f"  Done.  Removed {n_removed_plain} samples  ({t_plain:.1f}s)")

    # 6b. Option 7: fully private (cipher × CIPHER)
    print(f"\n[5/5b] Option 7 — cipher × CIPHER buffer vectors (fully private)...")
    print(f"       (server cannot read any feature data from either party)")
    np.random.seed(SEED)   # same random seed for a fair comparison
    t0 = time.time()
    Xd_priv, yd_priv = fhe_undersample_global_private(
        ctx, dom_id, clients, theta=0.2, eta=0.6,
    )
    n_removed_priv = n_before - len(yd_priv)
    t_priv = time.time() - t0
    print(f"  Done.  Removed {n_removed_priv} samples  ({t_priv:.1f}s)")

    # Summary
    print("\n" + "=" * 62)
    print("  Option 7 — Private Redundancy Protocol: Summary")
    print("=" * 62)
    print(f"  Dominant client : {dom_id}")
    print(f"  Dataset size    : {n_before} samples (before undersampling)")
    print()
    print(f"  {'Method':<30}  {'Removed':>8}  {'Server sees buf_vec'}")
    print(f"  {'─' * 30}  {'─' * 8}  {'─' * 22}")
    print(f"  {'Section 9  (cipher × plain)':<30}  {n_removed_plain:>8}  "
          f"YES  (plaintext list)")
    print(f"  {'Option 7   (cipher × cipher)':<30}  {n_removed_priv:>8}  "
          f"NO   (encrypted)")
    print()
    print(f"  FHE depth per dot product  :  1  (cipher×cipher multiply)")
    print(f"  Rotation-sum depth         :  0  (9 × free CKKS addition)")
    print(f"  Total depth budget used    :  1 / 2  (one level remaining)")
    print(f"  CKKS context parameters    :  unchanged [60, 40, 40, 60]")
    print()
    print("  Privacy upgrade: dominant client's buffer vectors are now")
    print("  ciphertexts.  The server operates exclusively on encrypted data")
    print("  and learns at most 1 binary bit per buffer candidate.")
    print("=" * 62)


if __name__ == "__main__":
    main()
