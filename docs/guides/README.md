# xai-channel-flow Guide Hub

This folder contains four coordinated documents derived from mining the xai-channel-flow source tree using the topology-first workflow:

| File | Audience | Contents |
| --- | --- | --- |
| [`topology.md`](./topology.md) | Everyone (start here) | Structural map (Mermaid) + entity inventory + dependency edges. The backbone for the other docs. |
| [`developer_guide.md`](./developer_guide.md) | Framework developers / contributors | Architecture, module tours, extension SOPs, debugging tips. |
| [`user_guide.md`](./user_guide.md) | End users / applied engineers | Installation, core API, typical workflows, FAQ. |
| [`tutorial.md`](./tutorial.md) | Learners at any level | 10-chapter walkthrough grounded in the repo's own numbered pipeline. |

> Upstream project: `https://github.com/KTH-FlowAI/XAI_turbulentchannel_3d_simplified` — **resolved externally** via `git ls-remote` against both org URLs (not found in-repo): `KTH-FlowAI` carries a `master` branch and a stale pull-request ref (`refs/github-services/pull/1/head`) that the `VinuesaLAB-AI` copy (this clone's `origin`) lacks, indicating `KTH-FlowAI` holds the original git history and `VinuesaLAB-AI` is a mirror. Both orgs' `main` branches point to the identical latest commit (`6c82482`), so this distinction affects provenance only, not content.
> Official docs: none — `README.md` is the only documentation upstream. This guide set is the first one produced for this repository.

## How these documents were produced

- Based on direct reading of `code/`, `code/configuration/`, `code/py_bin/`, and the existing `README.md`.
- Every claim is grounded in a concrete source path; unresolved facts are marked with `> TODO(doc-miner): ...`.
- `code/py_bin/py_packages/{shap,slicer,cloudpickle}/` are vendored third-party libraries (each ships its own `LICENSE.txt`) and are treated as out of scope for deep documentation, the same way the sibling `Diff-SPORT` repo's docs treat its vendored `libs/shap/`.

## Citation

This repository's own `README.md` and `LICENSE` carry **no citation information** — no DOI, no arXiv ID, no `CITATION.cff`. The canonical citation below was **resolved externally** (web search, not found in-repo) by matching this repo's code — specifically its percolation and coincidence-analysis scripts that compare SHAP-important regions against classical coherent structures (Q events, streaks, Chong vortices) — against the paper's abstract, which describes exactly that comparison:

> Cremades, A., Hoyas, S. & Vinuesa, R. "Classically studied coherent structures only paint a partial picture of wall-bounded turbulence." *Nature Communications* **16**, 10189 (2025). DOI: [10.1038/s41467-025-65199-9](https://doi.org/10.1038/s41467-025-65199-9). Preprint: [arXiv:2410.23189](https://arxiv.org/abs/2410.23189).

This is distinct from the earlier, simpler Cremades et al., *Nat. Commun.* 15, 3864 (2024) paper, which the sibling repository `Identifying-regions-of-importance-in-wall-bounded-turbulence-through-explainable-deep-learning` already documents.

> Recommendation: add a `CITATION.cff` (as `Diff-SPORT` already does) so this citation doesn't need to be sourced externally in future.

## Reading order suggestions

1. **New users** — start with `user_guide.md` §1–§5, then Chapters 1–4 of `tutorial.md`.
2. **Applied engineers** reproducing the paper's pipeline — jump to `user_guide.md` §19 (Minimal runnable templates) or the matching chapter of `tutorial.md`.
3. **Contributors** — read `developer_guide.md` end to end; cross-reference `user_guide.md` when a script's parameters are discussed.
