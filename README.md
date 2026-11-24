# BARG-CC Reliability Studio — V3.2 (IJSS 2026)

A runnable Streamlit source application adapted from the supplied HAMVCC Premium UI V3, with a redesigned Studio interface, sidebar workspace navigation, archive processing and interactive charts. The research basis is the supplied corrected 44-page manuscript, **BARG-CC: reliability-constrained information fusion with certified view-loss stability and failure diagnostics**, by Debmalya Ray and Prodipto Das. It is a research workbench, not a certified production system or a claim of journal acceptance.

## V3.2 — modern Streamlit layout API

- Migrates all chart, dataframe and image sizing calls to `width="stretch"` (or `width="content"` for intrinsic sizing). The shared Plotly helper accepts the new width parameter and retains custom styling and export controls.
- Requires Streamlit >=1.51,<2; startup checks the installed version and repairs incompatible environments. Existing V3.1 installations should install the updated requirements once, then restart.
- Preserves the numerical engine, evidence archive, themes, and V3.1 rendering fixes. See `VALIDATION.txt` for executed checks. No claim of live browser verification is made.

## V3.1 — clean rendering maintenance release

- Cluster summary and keyword rendering now uses explicit `if/else` blocks instead of a bare conditional expression returning a Streamlit DeltaGenerator. Both nonempty and empty keyword cases are tested.
- Evidence hash verification uses explicit success/error branches as well.
- Automatic expression rendering is disabled via project `.streamlit/config.toml` (`[runner] magicEnabled = false`) and the supplied launch scripts. Explicit Streamlit widgets and charts remain active. Restart the server after changing this configuration.
- All three Studio palettes, the document inspector fix, computation engine and archived evidence are preserved. No dependencies changed.
- 33 offline tests passed, including five new rendering regression checks. The enhanced `smoke_ui.py` checks for unexpected DeltaGenerator output after interactions; execute it with the installed Streamlit environment. Live Streamlit and Windows testing remain unverified in the build environment.

## Studio V3 presentation and interaction

- Sidebar **Appearance**: Aurora (indigo/teal), Ocean (blue/cyan), or Graphite (plum/copper). All use a light analytical canvas and dark navigation. Native Streamlit widgets retain a light theme; appearance is not a full dark-mode toggle.
- Local system font stack (Inter if installed, otherwise Segoe UI / Arial; Aptos Display if installed). No font downloads, embedded font files, or external CSS dependencies.
- Workspace headers, KPI cards, accessible focus outlines, reduced-motion support and responsive spacing. Plotly charts use matching palettes, hover information, zoom/pan and PNG export. Certified and observed-unchanged outcome colors remain consistent across palettes.
- Workspace 02: interactive exact view-loss curve, with selected removed mass indicated when a valid subset of views is chosen. This illustrates Eq. (22), not semantic performance.
- Workspace 05: live import / fit / audit status strip; side-by-side K selection and elbow diagnostics; expandable diagnostic table; filename hover in PCA. Changing appearance does not refit the clustering model. All previous downloads and numerical routines are retained.
- `studio_theme.py` contains presentation helpers only. Existing numerical engine, dependencies, V2 computation profiles, and V2.1 inspector fix are unchanged.
- Validation: offline regression tests and Plotly serialization; full Streamlit AppTest and browser/Windows rendering still require local verification. Run `python smoke_ui.py` after installation. A browser checklist is in `UI_CHECKLIST.md`.

## V2.1 compatibility hotfix

The document inspector uses separate label, hash and OCR status elements rather than multi-argument column `.write()` calls. No clustering, certificate, dependency or performance-profile changes. The smoke test now loads the demo archive, runs clustering, changes the inspected document and executes the one-refit audit.

## Windows quick start

Extract the entire ZIP (do not run from inside the ZIP). Open the extracted `BARG_CC_Streamlit_Studio_V3_2` folder and double-click `start_windows.bat`. It creates an isolated `.venv`, installs dependencies and starts Streamlit on http://127.0.0.1:8501. Python 3.11 or 3.12 is recommended. Initial dependency installation needs internet access. If 8501 is busy, launch manually with another port.

PowerShell from the extracted folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py --runner.magicEnabled=false --server.address 127.0.0.1 --server.port 8501
```

If using an already configured conda environment, use `python -m pip install -r requirements.txt` then `python -m streamlit run streamlit_app.py --runner.magicEnabled=false --server.address 127.0.0.1 --server.port 8501` (the batch launcher instead uses its own .venv). Module invocation avoids mismatched PATH/Streamlit launchers. Keep the terminal open. Ctrl+C stops the server. For Linux/macOS, install requirements in a virtual environment and use `bash start.sh`.

Optional OCR is disabled by default in the UI and batch command. To enable it, select the checkbox before import or pass `--ocr` in batch mode. Install the separate Tesseract OCR executable and put it on PATH. Without it, image features still work and OCR status reports the failure; the text view is excluded when no text is available. PDFs use pypdf text extraction (first 30 pages), not image rendering or OCR. Password-protected or malformed PDFs may be unassignable. Only upload trusted files; this local research parser is not an isolation sandbox for hostile documents. Uploaded content is processed in the server process and held in session memory. Use the Clear imported data and cached features button, or stop the server to release session records. The upload widget itself can still retain the selected file until cleared. Hosted deployments process files on the host, not the user's PC.

## Optimized computation (V2)

- Research (default): 10 K-means restarts, full silhouette, same feature/fit budget as V1. Quick: 3 restarts, a seeded shared sample of at most 600 rows for silhouette; if a candidate lacks cluster coverage, all candidates are scored on all rows. Quick may change K and labels; it is exploratory.
- When kappa=0 (or every row has a single observed view), the identical baseline fit and pairwise distances are reused. Otherwise both matched branches are fitted. Pairwise distances are reused across candidate K. Full distance matrices require up to about 36 MB for two float64 matrices at the 1,500-record upload limit, plus working memory.
- Distances to centroids use a two-dimensional distance matrix, avoiding an n × K × d broadcast intermediate. Exact certificate thresholds and assumptions remain unchanged.
- Session-scoped feature cache is keyed by archive SHA-256, OCR setting and seed; changing K or gate settings does not refit feature preprocessing. PCA and keywords are computed only after clustering. The cache is not shared across users or written to disk.
- Import and fit progress are displayed. The report records computation profile, fit count, OCR setting, selected K and elapsed time. Windows startup checks the requirements fingerprint and installed imports, skipping pip installation after a successful setup.

### Easiest workflow

1. Double-click `start_windows.bat`. Open http://127.0.0.1:8501 if a browser does not open automatically.
2. Workspace 05: import an archive (or the mixed-document demo). Leave OCR off for initial visual-only/text-file exploration; enable it if image text is needed.
3. Choose Quick for an initial run or Research for the previous full fitting budget. Keep availability-only unless you specifically want to examine bounded gating. Run clustering, then optionally audit view-loss stability.

### Local batch mode (no browser)

After installing requirements, double-click `run_demo_batch.bat` for a small demo, or use PowerShell:

```powershell
.\.venv\Scripts\python.exe run_batch.py "E:\Data\documents.zip" --profile Quick --k-min 2 --k-max 4 --dropout 0.1 --output results_quick
```

Add `--ocr` for image OCR, `--gated --floor 0.1` for bounded gating, or `--profile Research` for full-budget scoring. Use a new/empty output directory each time. Batch mode emits assignments, K diagnostics, excluded and skipped records, optional pointwise/aggregate view-loss audit, and JSON provenance.

### Performance audit

`performance_results.json` contains measured timings and parity checks on 1,000 synthetic records (K=2–8; three repetitions). To reproduce, extract the original ZIP separately and run `python benchmark_optimization.py --original-engine path/to/original/engine.py`. Timings are engine-only and do not imply improvements in semantic utility. Quick is not a replacement for full-budget manuscript experiments. Archived evidence is unchanged.

## Workspaces

| Workspace | Functionality |
|---|---|
| 01 Results | 12 curated archived evidence families, filterable tables, Plotly charts, full CSV/JSON artifact browser. Includes overall and affected-row coverage (never conflate them). |
| 02 Bounded gate lab | Three availability checkboxes, scores, floor/strength control, Jacobian, fixed-mask drift bound, exact view-loss drift and participation-floor tradeoff. |
| 03 Document features | Public Digits feature inventory and distributions; imported document features and text inspector. |
| 04 Diagnostics | Orthogonal coordinate invariance, exact global two-cluster enumeration of eight records, compositional fixed-mask certificate. |
| 05 Archive upload & clustering | ZIP or individual files, deduplication, optional OCR, availability-only/gated matched comparison, K sweep, silhouette, Davies–Bouldin, elbow, PCA, cluster keywords, search, inspector, seed stability, one-refit audit and exports. Try the built-in mixed-document demo. |
| 06 View-loss certification | Archived overall and event-conditioned results, new one-fold Digits simulation with unit blocks and frozen scores, per-record certificates, observed persistence and false-certificate checks. |
| 07 Research & provenance | Original manuscript PDF, original architecture image, evidence matrix and hash verification. |

## Manuscript alignment

- Eq. 2 / Propositions 1–3: masked softmax, absent weights zero, all-missing inputs rejected. A shared strength is capped using three views and the chosen floor. Availability-only (`kappa=0`) is the default; gating must be explicitly enabled.
- Proposition 4: `b = concat(sqrt(M_i * alpha_m) * e_m)`. The earlier HAMVCC document application used linear alpha weights; this implementation uses the square root as required by the corrected manuscript. Unnormalized `b` at kappa zero equals ordinary masked concatenation.
- Proposition 10: score, embedding and centre budgets compose under a fixed mask. The live demonstration uses synthetic bounded perturbations, not estimated deployment budgets.
- Propositions 12–14: `z = b/sqrt(M_i)`, with unit observed view embeddings; the exact loss distance is `sqrt(2*(1-sqrt(1-q)))`. A zero encoded block is flagged as unavailable for this app's feature model. One-refit certification checks all competitor margins. It is not arbitrary Lloyd convergence or global optimality.
- Proposition 11: all 127 unlabelled K=2 partitions of eight synthetic records are enumerated, including centroid refitting via the objective. This exponential routine is deliberately limited to small n.
- The source archive's `hamv_gate.py`, `mask_switch_certificate.py`, and `global_reoptimized_certificate.py` are retained as supplied. The app calls their applicable numerical routines via `engine.py`. The app's new simulations are not presented as replications of the full archived neural training. No trained checkpoints, specialist baselines or independent natural modalities are implied.

## Evidence and limitations

The archived `reference/results` folder is copied from the supplied corrected LaTeX source. The original manuscript reports 1,797 Digits records, 77.1% overall one-refit coverage at 10% simulated dropout, and zero certification among the 189 records that actually lost a view. These are archived results, not generated by the app at startup. The live Digits experiment uses train-fold-only preprocessing and transductive held-out-fold clustering; its seed and data split differ from the manuscript's five-fold run. No external label enters live K-means fitting; labels are used for stratified split or displayed in feature inventory only. For uploaded documents, both policies use the same processed blocks, K and seed; internal silhouette and partition ARI do not measure semantic utility. The image/text reliability proxies are uncalibrated. Unknown is not failure, and no observed false certificates is not a future-risk guarantee.

## Tests and local smoke check

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe smoke_ui.py
```

`VALIDATION.txt` records actual engine tests executed during packaging. The Streamlit browser and AppTest are not verified in the build environment because Streamlit was unavailable and could not be installed there. The optional `smoke_ui.py` is provided for the user's installed Streamlit environment. For manual testing: open each workspace, load the bundled mixed archive, run clustering and the one-refit audit, then visit workspace 06 and execute a Digits simulation. Check downloads. No remote deployment has been performed.
