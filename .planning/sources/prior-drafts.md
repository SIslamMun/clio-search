# Prior Drafts Index

All files treated as CONTEXT ONLY — paper starts fresh.

## Existing Documents

| File | Type | Description | Usable Content | Target Section |
|------|------|-------------|---------------|----------------|
| `intro/abstract.md` | Draft abstract | v0.5, 250 words, SC2026 framing | Key framing, 3-contribution structure | Abstract (rewrite) |
| `intro/introduction.md` | Draft intro | v0.5, full intro with 3-problem framing | Problem statements, citations, contributions | Section 1 (rewrite) |
| `background/background.md` | Draft background | Full §2: hybrid retrieval, numerical crisis, quantity-aware, HPC | Strong evidence + citations | Section 2 (rewrite) |
| `background/papers/01_prior_art_and_competitors.md` | Research notes | 16 papers with threat levels and differentiation | Differentiation arguments | Related work |
| `background/papers/02_gaps_and_evidence.md` | Research notes | Three failures + infrastructure papers | Gap validation | Background §2.6 |
| `background/papers/03_agentic_gaps_evidence.md` | Research notes | Evidence agents fail at data | Agent failure stats | Background §2.4 |
| `background/papers/04_agentic_science_papers.md` | Research notes | 4-act narrative: agents doing science | Motivation narrative | Introduction |
| `design/architecture.md` | Draft design | Full §3: system overview, SI conversion, pipeline, API | Technical spec, formulas, diagrams | Section 3 (rewrite) |
| `design/novelty.md` | Research notes | Core novelty, 3 problems, comparison table | Differentiation table | Related work |
| `design/hypothesis.md` | Research notes | 3 RQs, 3 hypotheses, motivation | RQ framing | Introduction |
| `design/figures.md` | Planning | 6 figures + 2 tables planned | Figure descriptions | All sections |
| `codebases/related_repos_analysis.md` | Research notes | 17 GitHub repos analyzed | Threat matrix | Related work |

## Key Arguments to Preserve

1. **The dimensional conversion argument**: kPa×10³=Pa vs string normalization — guaranteed correct by construction
2. **Nobody searches data**: OpenScholar/HiPerRAG/Context-1 all search papers, not HDF5/NetCDF/experimental data
3. **Three compounding failures**: dimensional blindness + formula opacity + storage fragmentation
4. **0.54 accuracy** on numbers (Numeracy Gap, EACL 2026) — key evidence stat
5. **32% ScienceAgentBench** success rate — agents fail at data, not reasoning

## Material NOT to Carry Forward

- "v0.5 FINAL" version labels
- Internal notes and threat level analysis
- Redundant comparison tables (consolidate to one)
