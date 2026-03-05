"""
Option 6 — Encrypted Gradient Aggregation (FHE-FedAvg)
=======================================================

Extends FHE-Flicker into a full Privacy-Preserving Federated Learning (PPFL)
framework. Rather than sharing plaintext gradients with the server, each client
trains locally, encrypts its linear-head gradient with CKKS, and sends
ciphertexts. The server aggregates ciphertexts using homomorphic addition
(zero multiplicative depth) and the key-holder applies the FedAvg update after
decryption.

Key design decisions
--------------------
1. Only the linear head (10×512 + 10 = 5,130 params) is encrypted — the ResNet
   backbone is frozen and contributes no gradients to be protected.
2. 5,130 > 4,096 slots (poly_modulus_degree=8192), so the gradient is split
   into two CKKS chunks: [0:4096] and [4096:5130].
3. CKKS addition consumes zero multiplicative levels, so the aggregation step
   uses the same [60, 40, 40, 60] context as the rest of FHE-Flicker without
   any parameter changes.

Usage
-----
    python option6_fhe_fedavg.py

Requirements
------------
    pip install torch torchvision tenseal numpy
"""

import copy
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

GRAD_CHUNK_SIZE = 4096          # CKKS slots per ciphertext (N/2 for poly_deg=8192)
N_HEAD_PARAMS   = 10 * 512 + 10 # 5130: weight (5120) + bias (10)
N_CLIENTS       = 3
DIRICHLET_ALPHA = 0.3           # lower → more imbalanced client distributions
FEDAVG_ROUNDS   = 5
LEARNING_RATE   = 1e-3
SEED            = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# CKKS context
# ─────────────────────────────────────────────────────────────────────────────

def setup_ckks_context():
    """
    Initialise a TenSEAL CKKS context shared across all clients and the server.

    Parameters chosen to match the main FHE-Flicker notebook:
      poly_modulus_degree = 8192   →  N/2 = 4096 CKKS slots, ≥128-bit security
      coeff_mod_bit_sizes           →  [60, 40, 40, 60]: supports depth-2 circuits
      global_scale = 2^40          →  ~12 decimal digits of floating-point precision

    Galois keys are generated to support ciphertext rotations (used by Option 7
    and by the FHE inference step in the main notebook; available here at no
    extra security cost).
    """
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    ctx.generate_galois_keys()
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
resnet.fc = nn.Identity()   # discard classification head; output is 512-dim
resnet.eval()
resnet = resnet.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Model definition
# ─────────────────────────────────────────────────────────────────────────────

class CIFARClassifier(nn.Module):
    """
    Frozen ResNet-18 backbone + trainable linear head.

    forward(x) : (B, 3, 224, 224) → backbone (frozen) → (B, 512)
                                  → head (trainable)   → (B, 10) logits

    Only head.weight and head.bias have requires_grad=True, so gradient
    vectors are small (5,130 params) and cheap to encrypt.
    """
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights="IMAGENET1K_V1")
        backbone.fc = nn.Identity()
        for p in backbone.parameters():
            p.requires_grad = False
        self.backbone = backbone
        self.head = nn.Linear(512, 10)

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)
        return self.head(x)


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
    """Dirichlet-sample per-client counts that sum to total."""
    proportions = np.random.dirichlet([alpha] * n_clients)
    counts = (proportions * total).astype(int)
    counts[-1] = total - counts[:-1].sum()   # fix rounding
    return np.maximum(counts, 1)             # every client gets ≥1 sample


def distribute_cifar_imbalanced(labels, n_clients, alpha=0.5, seed=None):
    """
    Distribute CIFAR-10 indices across clients using a Dirichlet imbalance model.

    For each class c, the number of samples going to each client is drawn from
    Dirichlet(alpha). The assignment order is rotated by (c*3) % n_clients so
    that different classes end up dominant on different clients.

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


def evaluate(model, loader):
    """Compute top-1 accuracy on a DataLoader."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(dim=1) == y).sum().item()
            total   += y.size(0)
    return correct / total if total > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Gradient packing and encryption helpers
# ─────────────────────────────────────────────────────────────────────────────

def flatten_head_gradients(local_model):
    """
    Extract and flatten the linear head's accumulated gradient to 1-D float32.

    head.weight.grad : shape (10, 512) → 5,120 elements
    head.bias.grad   : shape (10,)     →    10 elements
    Concatenated     :                  → 5,130 elements total
    """
    w_grad = local_model.head.weight.grad.detach().cpu().numpy().flatten()  # 5120
    b_grad = local_model.head.bias.grad.detach().cpu().numpy()              #   10
    return np.concatenate([w_grad, b_grad]).astype(np.float32)              # 5130


def encrypt_gradients(ctx, grad_flat):
    """
    Encrypt a flat gradient array as a list of CKKS-vector chunks.

    Chunking overcomes the 4096-slot limit for our 5130-parameter head:
      chunk 0 → grad_flat[0    : 4096]  (4096 elements, fully packed)
      chunk 1 → grad_flat[4096 : 5130]  (1034 elements; TenSEAL zero-pads)

    The server receives 2 opaque CKKSVector ciphertexts per client.
    It cannot distinguish a zero-padded slot from a real gradient element.
    """
    chunks = [grad_flat[i : i + GRAD_CHUNK_SIZE]
              for i in range(0, len(grad_flat), GRAD_CHUNK_SIZE)]
    return [ts.ckks_vector(ctx, c.tolist()) for c in chunks]


def aggregate_encrypted_gradients(enc_grads_per_client):
    """
    Server-side aggregation: sum each gradient chunk across all clients.

    enc_grads_per_client : list[list[CKKSVector]]
      Outer list — one entry per client.
      Inner list — chunked gradient ciphertexts (2 chunks for a 5130-param head).

    Returns: list[CKKSVector]  — the element-wise sum of all clients' chunks,
    still fully encrypted.

    Depth consumed: 0.
      CKKS addition does NOT consume multiplicative levels. It is purely
      polynomial ring addition: c_sum = c_0 + c_1 + … + c_{n-1}. Noise
      grows as O(sqrt(n_clients)) but stays well within the budget set by
      the 40-bit intermediate primes at scale = 2^40.
    """
    n_chunks = len(enc_grads_per_client[0])
    aggregated = []
    for k in range(n_chunks):
        total = enc_grads_per_client[0][k]
        for client_grads in enc_grads_per_client[1:]:
            total = total + client_grads[k]   # homomorphic addition, depth = 0
        aggregated.append(total)
    return aggregated


def decrypt_and_apply_update(global_model, agg_chunks, n_clients, lr=LEARNING_RATE):
    """
    Key-holder: decrypt the aggregated gradient and apply the FedAvg update.

    FedAvg rule:   θ ← θ − lr · (1/n) · Σᵢ grad_i

    agg_chunks holds Σᵢ grad_i (the sum). We divide by n_clients to get the
    mean gradient before applying the update.

    CKKS approximation note: decrypted values carry ~2^{-40} relative error
    (scale = 2^40). For gradient-based optimisation this is completely
    negligible — typical SGD tolerates noise several orders of magnitude larger
    from mini-batch sampling alone.
    """
    flat = []
    for chunk in agg_chunks:
        flat.extend(chunk.decrypt())
    flat = np.array(flat[:N_HEAD_PARAMS], dtype=np.float32)   # trim zero-padding

    w_grad = flat[:10 * 512].reshape(10, 512)
    b_grad = flat[10 * 512:]

    with torch.no_grad():
        global_model.head.weight -= lr * torch.tensor(
            w_grad / n_clients, dtype=torch.float32, device=device)
        global_model.head.bias   -= lr * torch.tensor(
            b_grad / n_clients, dtype=torch.float32, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# Local training
# ─────────────────────────────────────────────────────────────────────────────

def local_train_one_epoch(global_model, loader, criterion):
    """
    Client-side: deep-copy the global model and train the linear head for one
    epoch on the client's local data.

    Design decisions
    ----------------
    * Only head parameters are trained — backbone is frozen — so the gradient
      vector is small enough (5,130 params) to encrypt with 2 CKKS chunks.
      Encrypting backbone gradients would require ~2,700 chunks and dominate
      runtime.
    * Gradients are accumulated over all batches and normalised by the total
      sample count. This is equivalent to one gradient descent step on the
      full local dataset (not stochastic), reducing variance and improving
      aggregation fidelity under encryption.

    Returns the trained local_model (caller reads .grad tensors from it).
    """
    local_model = copy.deepcopy(global_model)
    local_model.train()
    for p in local_model.backbone.parameters():
        p.requires_grad = False
    local_model.head.zero_grad()

    total_samples = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss  = criterion(local_model(x), y)
        loss.backward()          # accumulates into .grad (not zeroed between batches)
        total_samples += x.size(0)

    # Normalise so the gradient represents a per-sample mean, not a raw sum
    if total_samples > 0:
        local_model.head.weight.grad.div_(total_samples)
        local_model.head.bias.grad.div_(total_samples)

    return local_model


# ─────────────────────────────────────────────────────────────────────────────
# FHE-FedAvg training loop
# ─────────────────────────────────────────────────────────────────────────────

def run_fhe_fedavg(ctx, clients, test_loader, global_model=None,
                   rounds=FEDAVG_ROUNDS, lr=LEARNING_RATE):
    """
    Privacy-Preserving Federated Learning with CKKS-Encrypted Gradients.

    Per-round protocol
    ------------------
    1. Server broadcasts the current global model weights (plaintext — standard
       in all FL frameworks, including FedAvg).
    2. Each client trains locally for one epoch (data never leaves the client).
    3. Client encrypts its gradient:
         grad_flat (5130 float32) → encrypt_gradients() → 2 CKKSVector chunks
         Server receives 2 opaque ciphertexts per client.
    4. Server aggregates ciphertexts:
         agg_k = Σᵢ enc_grad_i_chunk_k   (homomorphic addition, depth = 0)
    5. Key-holder decrypts → agg_k → applies FedAvg update:
         θ ← θ − lr · (1/n) · Σᵢ grad_i

    What the server observes
    ------------------------
    Only CKKS ciphertexts. Without the secret key the server cannot distinguish
    the encryption of [0.01, −0.03, …] from a freshly sampled random polynomial.

    Simulation note
    ---------------
    In this demo a single TenSEAL context (shared secret key) is used so we can
    verify results by decryption. In a real deployment:
      * The server holds only the public evaluation key.
      * Clients each hold the secret key (or threshold CKKS shares of it).
      * The server never calls .decrypt() — it ships the aggregate ciphertext
        back to the clients for joint decryption.

    Parameters
    ----------
    ctx          : TenSEAL CKKS context
    clients      : dict  {client_id: {"X": Subset, "y": np.ndarray}}
    test_loader  : DataLoader for the held-out test set
    global_model : CIFARClassifier (freshly initialised if None)
    rounds       : number of FL rounds
    lr           : learning rate for FedAvg update
    """
    if global_model is None:
        global_model = CIFARClassifier().to(device)
        print("  Initialised fresh global model.")

    criterion = nn.CrossEntropyLoss()
    n_clients  = len(clients)
    acc_history = []

    for rnd in range(rounds):
        t0 = time.time()
        print(f"\n{'─' * 58}")
        print(f"  FHE-FedAvg  Round {rnd}  ({n_clients} clients)")
        print(f"{'─' * 58}")

        enc_grads_per_client = []

        for cid, client in clients.items():
            loader = DataLoader(
                client["X"], batch_size=128, shuffle=True,
                num_workers=0, collate_fn=safe_collate,
            )
            local_model = local_train_one_epoch(global_model, loader, criterion)
            grad_flat   = flatten_head_gradients(local_model)    # (5130,) float32
            enc_grads   = encrypt_gradients(ctx, grad_flat)      # list of 2 CKKSVectors

            enc_grads_per_client.append(enc_grads)
            print(f"  Client {cid}: {N_HEAD_PARAMS} params → "
                  f"{len(enc_grads)} CKKS chunk(s) sent (encrypted)")

        # Server: purely ciphertext addition, depth = 0
        agg_chunks = aggregate_encrypted_gradients(enc_grads_per_client)
        print(f"  Server: aggregated {n_clients} clients "
              f"(homomorphic addition, depth = 0)")

        # Key-holder: decrypt and update
        decrypt_and_apply_update(global_model, agg_chunks, n_clients, lr)

        acc = evaluate(global_model, test_loader)
        acc_history.append(acc)
        print(f"  Test accuracy: {acc:.4f}   ({time.time() - t0:.1f}s)")

    return global_model, acc_history


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Option 6 — FHE-FedAvg: Encrypted Gradient Aggregation")
    print("=" * 60)

    # 1. CKKS context
    print("\n[1/5] Setting up CKKS context...")
    ctx = setup_ckks_context()
    print("  poly_modulus_degree = 8192  (4096 slots per ciphertext)")
    print("  coeff_mod_bit_sizes = [60, 40, 40, 60]  (depth-2 budget)")
    print("  global_scale        = 2^40  (~12 decimal digits precision)")
    print("  Security            ≥ 128-bit  (HE-Standard)")
    print(f"  Device              = {device}")

    # 2. Load CIFAR-10
    print("\n[2/5] Loading CIFAR-10...")
    train_dataset = dset.CIFAR10(
        root="./data", train=True, download=True, transform=transform_resnet
    )
    test_dataset = dset.CIFAR10(
        root="./data", train=False, download=True, transform=transform_resnet
    )
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False,
        num_workers=0, collate_fn=safe_collate,
    )
    all_labels = np.array(train_dataset.targets)
    print(f"  Train: {len(train_dataset)} samples | Test: {len(test_dataset)} samples")

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

    # 4. Gradient encryption round-trip verification
    print("\n[4/5] Verifying gradient encryption round-trip...")
    tmp_model = CIFARClassifier().to(device)
    tmp_loader = DataLoader(
        Subset(test_dataset, list(range(256))), batch_size=256,
        shuffle=False, num_workers=0, collate_fn=safe_collate,
    )
    criterion_v = nn.CrossEntropyLoss()
    tmp_model.head.zero_grad()
    x_v, y_v = next(iter(tmp_loader))
    x_v, y_v = x_v.to(device), y_v.to(device)
    criterion_v(tmp_model(x_v), y_v).backward()
    tmp_model.head.weight.grad.div_(x_v.size(0))
    tmp_model.head.bias.grad.div_(x_v.size(0))

    plain_grad  = flatten_head_gradients(tmp_model)
    enc_chunks  = encrypt_gradients(ctx, plain_grad)
    agg_single  = aggregate_encrypted_gradients([enc_chunks])   # identity (n=1)

    flat_dec = []
    for chunk in agg_single:
        flat_dec.extend(chunk.decrypt())
    flat_dec = np.array(flat_dec[:N_HEAD_PARAMS], dtype=np.float32)

    max_err = float(np.max(np.abs(plain_grad - flat_dec)))
    print(f"  Max absolute error (encrypt → aggregate → decrypt): {max_err:.2e}")
    print(f"  Expected: ~1e-7 for CKKS scale = 2^40  (negligible for SGD)")
    print(f"  Plaintext gradient ‖g‖ = {np.linalg.norm(plain_grad):.4f}")
    print(f"  Decrypted gradient ‖g‖ = {np.linalg.norm(flat_dec):.4f}")

    # 5. Run FHE-FedAvg
    print(f"\n[5/5] Running FHE-FedAvg ({FEDAVG_ROUNDS} rounds)...")
    final_model, acc_history = run_fhe_fedavg(
        ctx, clients, test_loader, global_model=None,
        rounds=FEDAVG_ROUNDS, lr=LEARNING_RATE,
    )

    # Summary
    print("\n" + "=" * 60)
    print("  FHE-FedAvg — Summary")
    print("=" * 60)
    for rnd, acc in enumerate(acc_history):
        print(f"  Round {rnd}: test accuracy = {acc:.4f}")
    print(f"\n  Final test accuracy : {acc_history[-1]:.4f}")
    print(f"\n  Privacy guarantee:")
    print(f"    Server received  : {N_CLIENTS * 2} CKKS ciphertexts per round")
    print(f"    Server saw       : 0 plaintext gradient values")
    print(f"    Aggregation depth: 0  (homomorphic addition only)")
    print(f"    CKKS context     : unchanged [60, 40, 40, 60]")
    print("=" * 60)


if __name__ == "__main__":
    main()
