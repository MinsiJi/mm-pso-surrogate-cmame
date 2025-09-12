# CMAME Code Release — MM‑PSO & Surrogate

This repository hosts four Python scripts used in the paper (submission to *CMAME*): MM‑PSO optimizer, surrogate model, three‑algorithm comparison, and ablation experiments.

> **Note**: File names have been normalized (no spaces) for cross‑platform compatibility.

## Repository Layout
```
.
├─ scripts/
│  ├─ mm_pso.py
│  ├─ surrogate_model.py
│  ├─ algorithm_comparison.py
│  └─ ablation.py
├─ data/           # place small demo/config files here (kept empty by default)
├─ results/        # results written here by scripts
├─ tests/          # optional tests
├─ requirements.txt
├─ CITATION.cff
├─ LICENSE (MIT)
└─ .gitignore
```

## Quickstart (English)
1. **Create environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate    # (Windows) .venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. **Run examples**
   ```bash
   # Hybrid PSO main run (adjust params inside the script)
   python scripts/mm_pso.py

   # Train/validate the surrogate model
   python scripts/surrogate_model.py

   # PSO vs GA vs MM-PSO comparison
   python scripts/algorithm_comparison.py

   # Ablation experiments (Surrogate-off, GP-off, DP-off)
   python scripts/ablation.py
   ```
3. **Reproducibility**
   - Set a global random seed inside each script if you need bit‑wise identical runs.
   - Heavy raw data (e.g., large CFD logs) are not tracked here; include minimal demo data in `data/`.

## 快速开始（中文）
1. **创建运行环境**
   ```bash
   python -m venv .venv
   source .venv/bin/activate    # (Windows) .venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. **运行脚本**
   ```bash
   python scripts/mm_pso.py
   python scripts/surrogate_model.py
   python scripts/algorithm_comparison.py
   python scripts/ablation.py
   ```
3. **可复现性建议**
   - 如需严格复现，请在脚本内统一设置随机种子。
   - 体量较大的原始数据不建议直接托管在仓库；可在 `data/` 中提供最小可运行示例。

## Data & Code Availability (template for the paper)
**Code:** Publicly available at the project GitHub repository (URL to be added after repository creation) and archived on Zenodo with a citable DOI (to be added).  
**Data:** A minimal example dataset and configuration files are included in the `data/` folder. Large intermediate files and raw logs produced during high‑fidelity simulations will be shared upon reasonable request due to storage limits.

> You can paste this paragraph directly into the *CMAME* Data Availability Statement and update the GitHub URL and DOI.

## How to cite
See `CITATION.cff`. After you create the GitHub repository and Zenodo archive, update:
- `repository-code:` with your GitHub URL
- `doi:` with the Zenodo DOI

## License
MIT (see `LICENSE`). You can switch to a different license if required by your project or sponsor.

---

**Next steps to publish on GitHub** (summary):
1. Create a new repository on GitHub (empty, without README).
2. Initialize locally and push:
   ```bash
   git init
   git add .
   git commit -m "Initial CMAME code release (v1.0.0)"
   git branch -M main
   git remote add origin <YOUR_GITHUB_REPO_SSH_OR_HTTPS_URL>
   git push -u origin main
   ```
3. Create a **Release** (e.g., `v1.0.0`) on GitHub. If you need a DOI, connect the repo to **Zenodo**, then make a release; Zenodo will mint a DOI.
