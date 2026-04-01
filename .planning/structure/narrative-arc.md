# Narrative Arc

## The Story in Three Acts

### Act 1: The Problem (§1-2)
**Reader experience**: "I didn't realize retrieval was this broken for scientific data."

AI agents are revolutionizing science — running experiments, writing papers, orchestrating cross-facility workflows. But they can't find data. The best agent scores 32% on science tasks because it fails at data retrieval, not reasoning. The root cause: retrieval treats "200 kPa" and "200000 Pa" as unrelated strings, can't match formulas across notations, and can't search across the storage backends where HPC data actually lives.

**Key emotional beat**: The motivating example — 3 documents, 3 backends, 3 unit representations. Every existing system finds at most 1. We find all 3.

### Act 2: The Solution (§3-4)
**Reader experience**: "This is surprisingly straightforward and principled."

Science-aware operators. Not a new neural architecture. Not a fine-tuned model. Arithmetic conversion: 200 × 1000 = 200000. Guaranteed correct by construction. Formula normalization. Federated search with capability negotiation. These operators execute as parallel branches alongside standard BM25 + vector retrieval — they complement, not replace.

**Key emotional beat**: The simplicity. String normalization tries to learn "kilopascal = pascal" and fails. Embeddings try to learn number similarity and get 0.54. We multiply by 1000. Done.

### Act 3: The Evidence (§5-6)
**Reader experience**: "The numbers back it up — dimensional conversion recovers results every other system misses."

Evaluation on scientific retrieval tasks. Dimensional conversion recovers X% of false negatives missed by all baselines. Formula matching adds Y%. Federated search discovers results hidden in other backends. Ablation shows each component contributes independently.

**Key emotional beat**: The comparison table showing all five novel capabilities are unique to our system — every other row empty for prior work.

## Reader Journey

```
"I know retrieval works"
  ↓ (Numbers are broken — 0.54 accuracy)
"Wait, numbers don't work in embeddings?"
  ↓ (Units are worse — string normalization can't cross prefixes)
"That's a real problem for scientific data"
  ↓ (Motivating example: 3 docs, 3 backends, 3 units)
"Okay, so how do you fix it?"
  ↓ (Arithmetic. Multiply by 1000. Guaranteed.)
"That's... elegant. What about formulas?"
  ↓ (Normalize and match — unified with measurement operators)
"And the federated part?"
  ↓ (Connector architecture, capability negotiation)
"Does it actually work?"
  ↓ (Evaluation: X% recall improvement, Y% F1 gain)
"This should exist for HPC."
```

## Tone

Direct. Precise. No hedging. This is a systems paper for HPC people — they want to know what it does, how it works, and whether the numbers support it. No "we believe" or "it could be argued." State the problem, state the solution, show the results.
