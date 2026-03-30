from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import linalg


SHARD_MAGIC = 20240520
SHARD_VERSION = 1
SHARD_HEADER_INTS = 256
VOCAB_SIZE = 1024
DATASET_PATH = Path("data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin")
RESULTS_DIR = Path("analysis/results")


def load_tokens(path: Path) -> np.ndarray:
    header_bytes = SHARD_HEADER_INTS * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(path, dtype="<i4", count=SHARD_HEADER_INTS)
    if header.size != SHARD_HEADER_INTS:
        raise ValueError(f"Unexpected shard header length for {path}")
    if int(header[0]) != SHARD_MAGIC or int(header[1]) != SHARD_VERSION:
        raise ValueError(f"Unexpected shard header for {path}")

    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if path.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {path}: expected {expected_size} bytes")

    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens.size != num_tokens:
        raise ValueError(f"Short read for {path}")
    return tokens.astype(np.uint16, copy=False)


def describe_tokens(tokens: np.ndarray, vocab_size: int) -> dict[str, float | int]:
    unique_tokens = np.unique(tokens)
    return {
        "num_tokens": int(tokens.size),
        "dtype": str(tokens.dtype),
        "min_token": int(tokens.min(initial=0)),
        "max_token": int(tokens.max(initial=0)),
        "num_unique_tokens": int(unique_tokens.size),
        "vocab_coverage": float(unique_tokens.size / vocab_size),
    }


def compute_unigram_counts(tokens: np.ndarray, vocab_size: int) -> np.ndarray:
    return np.bincount(tokens, minlength=vocab_size).astype(np.int64, copy=False)


def compute_bigram_matrix(tokens: np.ndarray, vocab_size: int) -> np.ndarray:
    matrix = np.zeros((vocab_size, vocab_size), dtype=np.int64)
    np.add.at(matrix, (tokens[:-1], tokens[1:]), 1)
    return matrix


def entropy_from_counts(counts: np.ndarray) -> float:
    total = counts.sum(dtype=np.float64)
    if total <= 0:
        return 0.0
    probs = counts.astype(np.float64, copy=False) / total
    nonzero = probs > 0.0
    return float(-np.sum(probs[nonzero] * np.log(probs[nonzero]), dtype=np.float64))


def compute_mutual_information(tokens: np.ndarray, vocab_size: int, k_max: int = 200) -> np.ndarray:
    unigram_probs = compute_unigram_counts(tokens, vocab_size).astype(np.float64, copy=False)
    unigram_probs /= unigram_probs.sum(dtype=np.float64)
    independent_probs = unigram_probs[:, None] * unigram_probs[None, :]

    mutual_information = np.zeros(k_max, dtype=np.float64)
    matrix = np.zeros((vocab_size, vocab_size), dtype=np.int64)
    for k in range(1, k_max + 1):
        matrix.fill(0)
        np.add.at(matrix, (tokens[:-k], tokens[k:]), 1)
        row_idx, col_idx = np.nonzero(matrix)
        joint = matrix[row_idx, col_idx].astype(np.float64, copy=False) / float(tokens.size - k)
        mutual_information[k - 1] = np.sum(
            joint * np.log(joint / independent_probs[row_idx, col_idx]),
            dtype=np.float64,
        )
        if k == 1 or k % 20 == 0 or k == k_max:
            print(f"mutual_info_progress={k}/{k_max}", flush=True)
    return mutual_information


def compute_zipf_exponent(counts: np.ndarray) -> float:
    flat_counts = np.ravel(counts)
    positive_counts = flat_counts[flat_counts > 0]
    if positive_counts.size == 0:
        return 0.0
    sorted_counts = np.sort(positive_counts)[::-1].astype(np.float64, copy=False)
    ranks = np.arange(1, sorted_counts.size + 1, dtype=np.float64)
    slope, _ = np.polyfit(np.log(ranks), np.log(sorted_counts), 1)
    return float(abs(slope))


def compute_transfer_spectrum(bigram_matrix: np.ndarray, top_k: int = 50) -> tuple[np.ndarray, float, float]:
    transition = np.zeros_like(bigram_matrix, dtype=np.float64)
    row_sums = bigram_matrix.sum(axis=1, keepdims=True, dtype=np.float64)
    nonzero_rows = row_sums[:, 0] > 0.0
    transition[nonzero_rows] = bigram_matrix[nonzero_rows] / row_sums[nonzero_rows]

    eigenvalues = linalg.eig(transition, check_finite=False)[0]
    order = np.argsort(np.abs(eigenvalues))[::-1]
    top_eigenvalues = eigenvalues[order][:top_k]

    if top_eigenvalues.size < 2:
        return top_eigenvalues, float("nan"), float("nan")

    lambda2_abs = float(np.abs(top_eigenvalues[1]))
    gap = float(1.0 - lambda2_abs)
    if lambda2_abs <= 0.0:
        correlation_length = 0.0
    elif np.isclose(lambda2_abs, 1.0):
        correlation_length = float("inf")
    else:
        correlation_length = float(-1.0 / np.log(lambda2_abs))
    return top_eigenvalues, gap, correlation_length


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual = np.sum((y - fitted) ** 2, dtype=np.float64)
    total = np.sum((y - y.mean()) ** 2, dtype=np.float64)
    r_squared = 1.0 if np.isclose(total, 0.0) else 1.0 - residual / total
    return float(slope), float(intercept), float(r_squared)


def nats_to_bits(value: float) -> float:
    return float(value / np.log(2.0))


def build_summary(
    h1_nats: float,
    h2_nats: float,
    delta_h1_nats: float,
    delta_h2_nats: float,
    mutual_information: np.ndarray,
    alpha1: float,
    alpha2: float,
    eigenvalues: np.ndarray,
    gap: float,
    correlation_length: float,
) -> str:
    ks = np.arange(1, mutual_information.size + 1, dtype=np.float64)
    positive_mask = mutual_information > 0.0
    log_mutual_information = np.log(mutual_information[positive_mask])

    power_slope, _, power_r2 = linear_fit(np.log(ks[positive_mask]), log_mutual_information)
    exp_slope, _, exp_r2 = linear_fit(ks[positive_mask], log_mutual_information)
    power_alpha = float(-power_slope)
    exp_decay_rate = float(-exp_slope)
    decay_model = "power_law" if power_r2 >= exp_r2 else "exponential"

    lambda1_abs = float(np.abs(eigenvalues[0])) if eigenvalues.size > 0 else float("nan")
    lambda2_abs = float(np.abs(eigenvalues[1])) if eigenvalues.size > 1 else float("nan")
    lambda3_abs = float(np.abs(eigenvalues[2])) if eigenvalues.size > 2 else float("nan")

    lines = [
        "FineWeb measurement summary",
        f"dataset_path={DATASET_PATH}",
        "",
        "Entropy (nats / bits)",
        f"H(1)={h1_nats:.6f} / {nats_to_bits(h1_nats):.6f}",
        f"H(2)={h2_nats:.6f} / {nats_to_bits(h2_nats):.6f}",
        f"DeltaH(1)={delta_h1_nats:.6f} / {nats_to_bits(delta_h1_nats):.6f}",
        f"DeltaH(2)={delta_h2_nats:.6f} / {nats_to_bits(delta_h2_nats):.6f}",
        "",
        "Mutual information decay",
        f"I(1)={mutual_information[0]:.6f}",
        f"I(10)={mutual_information[9]:.6f}",
        f"I(50)={mutual_information[49]:.6f}",
        f"I(100)={mutual_information[99]:.6f}",
        f"power_law_alpha={power_alpha:.6f}",
        f"power_law_r2={power_r2:.6f}",
        f"exponential_decay_rate={exp_decay_rate:.6f}",
        f"exponential_r2={exp_r2:.6f}",
        f"preferred_decay_model={decay_model}",
        "",
        "Zipf exponents",
        f"alpha1={alpha1:.6f}",
        f"alpha2={alpha2:.6f}",
        "",
        "Transfer matrix spectrum",
        f"|lambda1|={lambda1_abs:.6f}",
        f"|lambda2|={lambda2_abs:.6f}",
        f"|lambda3|={lambda3_abs:.6f}",
        f"spectral_gap={gap:.6f}",
        f"correlation_length_xi={correlation_length:.6f}",
        "",
        "Conclusion table",
        "measure | value | implication",
        f"H(1) | {h1_nats:.6f} nats | token frequencies are non-uniform, so unigram structure is worth preserving in embeddings or priors.",
        f"DeltaH(2) | {delta_h2_nats:.6f} nats | adjacent-token dependence is substantial, so local-context modeling remains important.",
        f"MI decay | {decay_model} | decay shape indicates whether long-range structure is better modeled as scale-free or short-memory.",
        f"Zipf alpha1 | {alpha1:.6f} | vocabulary usage is heavy-tailed, so capacity allocation should favor a small set of frequent tokens.",
        f"Zipf alpha2 | {alpha2:.6f} | bigram usage is also heavy-tailed, supporting sparse or low-rank transition structure.",
        f"Transfer spectrum | gap={gap:.6f}, xi={correlation_length:.6f} | the dominant transition modes quantify effective mixing speed and usable context length.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tokens = load_tokens(DATASET_PATH)
    stats = describe_tokens(tokens, VOCAB_SIZE)
    unigram_counts = compute_unigram_counts(tokens, VOCAB_SIZE)
    bigram_matrix = compute_bigram_matrix(tokens, VOCAB_SIZE)
    h1_nats = entropy_from_counts(unigram_counts)
    h2_nats = entropy_from_counts(bigram_matrix)
    delta_h1_nats = h1_nats
    delta_h2_nats = h2_nats - h1_nats
    mutual_information = compute_mutual_information(tokens, VOCAB_SIZE, k_max=200)
    alpha1 = compute_zipf_exponent(unigram_counts)
    alpha2 = compute_zipf_exponent(bigram_matrix)
    eigenvalues, gap, correlation_length = compute_transfer_spectrum(bigram_matrix, top_k=50)
    summary = build_summary(
        h1_nats=h1_nats,
        h2_nats=h2_nats,
        delta_h1_nats=delta_h1_nats,
        delta_h2_nats=delta_h2_nats,
        mutual_information=mutual_information,
        alpha1=alpha1,
        alpha2=alpha2,
        eigenvalues=eigenvalues,
        gap=gap,
        correlation_length=correlation_length,
    )

    np.savez(
        RESULTS_DIR / "entropy.npz",
        unigram_counts=unigram_counts,
        bigram_matrix=bigram_matrix,
        h1_nats=np.array(h1_nats, dtype=np.float64),
        h2_nats=np.array(h2_nats, dtype=np.float64),
        delta_h1_nats=np.array(delta_h1_nats, dtype=np.float64),
        delta_h2_nats=np.array(delta_h2_nats, dtype=np.float64),
    )
    np.savez(
        RESULTS_DIR / "mutual_info.npz",
        ks=np.arange(1, mutual_information.size + 1, dtype=np.int32),
        mutual_information_nats=mutual_information,
    )
    np.savez(
        RESULTS_DIR / "zipf.npz",
        alpha1=np.array(alpha1, dtype=np.float64),
        alpha2=np.array(alpha2, dtype=np.float64),
    )
    np.savez(
        RESULTS_DIR / "spectrum.npz",
        eigenvalues=eigenvalues,
        lambda_abs=np.abs(eigenvalues),
        spectral_gap=np.array(gap, dtype=np.float64),
        correlation_length=np.array(correlation_length, dtype=np.float64),
    )
    (RESULTS_DIR / "summary.txt").write_text(summary, encoding="utf-8")

    print(f"Loaded tokens from {DATASET_PATH}")
    print(f"results_dir={RESULTS_DIR}")
    print(f"num_tokens={stats['num_tokens']}")
    print(f"dtype={stats['dtype']}")
    print(f"min_token={stats['min_token']}")
    print(f"max_token={stats['max_token']}")
    print(f"num_unique_tokens={stats['num_unique_tokens']}")
    print(f"vocab_size={VOCAB_SIZE}")
    print(f"vocab_coverage={stats['vocab_coverage']:.6f}")
    print(f"h1_nats={h1_nats:.6f}")
    print(f"h2_nats={h2_nats:.6f}")
    print(f"delta_h1_nats={delta_h1_nats:.6f}")
    print(f"delta_h2_nats={delta_h2_nats:.6f}")
    print(f"saved_entropy={RESULTS_DIR / 'entropy.npz'}")
    for k in (1, 10, 50, 100, 200):
        print(f"mutual_info_k{k}_nats={mutual_information[k - 1]:.6f}")
    print(f"saved_mutual_info={RESULTS_DIR / 'mutual_info.npz'}")
    print(f"alpha1={alpha1:.6f}")
    print(f"alpha2={alpha2:.6f}")
    print(f"saved_zipf={RESULTS_DIR / 'zipf.npz'}")
    print(f"lambda1_abs={np.abs(eigenvalues[0]):.6f}")
    print(f"lambda2_abs={np.abs(eigenvalues[1]):.6f}")
    print(f"lambda3_abs={np.abs(eigenvalues[2]):.6f}")
    print(f"spectral_gap={gap:.6f}")
    print(f"correlation_length={correlation_length:.6f}")
    print(f"saved_spectrum={RESULTS_DIR / 'spectrum.npz'}")
    print(f"saved_summary={RESULTS_DIR / 'summary.txt'}")


if __name__ == "__main__":
    main()
