# BARG-CC / International Journal of Systems Science — corrected research package

23 November 2026. Prepared from the supplied BARG-CC IJSS LaTeX release (Figure 1 revision), preserving archived experiments and introducing a *descriptive, deterministic reanalysis* of event-conditioned view-loss certification. Not all remaining scientific or publisher-policy issues can be resolved from the available materials.

## Main files
- `main.tex` -> compiled `main.pdf` (44 pages)
- `supplement.tex` -> compiled `supplement.pdf` (25 pages)
- `main_review.pdf` (double-spaced, line numbered)
- `main_anonymous_review.pdf` (optional anonymized review)
- `supplement_anonymous.pdf` (optional anonymized supplement)
- `submission/COVER_LETTER_IJSS.pdf` (author approval required)
- `submission/AUTHOR_TITLE_PAGE.pdf`
- `submission/SCIENTIFIC_AND_SUBMISSION_AUDIT.pdf`
- `submission/AUTHOR_SIGNOFF_AND_ZERO_APC_CHECKLIST.pdf`
- `results/conditional_coverage_audit/conditional_coverage.csv` and figure
- `reproducibility/audit_conditional_coverage.py`
- `verification/FINAL_REVISED_TESTS.txt` (64 passed locally)

## Scientific contribution and boundaries
The fixed-mask mathematical statements and changed-mask view-loss and centroid-refit certificates are conditional, not deployed guarantees. The global-optimality condition requires exhaustive small-n partition evaluation. At 10% view-loss in Digits-derived views, 77.1% of all records satisfy the one-refit sufficient condition, but ZERO out of 189 records actually losing a view are certified. Observed 96.8% persistence in dropped-view records is *not* certified. The new conditional-coverage results are calculated from previously archived pointwise CSVs; not an independent trial. The historical neural study has partial reproducibility; native independent-modalities and strong specialist official-code comparisons remain incomplete.

## Build
`cd BARG_CC_IJSS_Corrected_LaTeX && bash build.sh`

Tests: `python -m unittest discover -s reproducibility -p 'test_*.py'`

Reaggregate the archived view-loss data: `python reproducibility/audit_conditional_coverage.py`

Python dependencies are recorded in requirements-repro.txt. Read legacy verification reports as earlier-stage evidence; the new executed tests and updated compile logs are in verification/.

## Journal fees, fit and final author sign-off
Publisher journal information (checked 23 November 2026):
https://www.tandfonline.com/journals/tsys20/about-this-journal
The non-OA SUBSCRIPTION route is stated to have NO APC. Optional Open Select may require an APC. The journal reports a 20% acceptance rate; this is journal-wide historical information, not a forecast for this paper. Other potential terms/optional services should be checked at filing. Official instructions page was blocked by the web source's access controls; therefore exact portal-specific acceptance of format is NOT independently certified.

Taylor & Francis AI guidance:
https://authorservices.taylorandfrancis.com/editorial-policies/using-ai-in-your-research-and-manuscript-preparations/
https://authorservices.taylorandfrancis.com/editorial-policies/images-and-figures/
The manuscript discloses AI assistance transparently. Both authors must verify the full draft timeline, Figure 1 provenance, any relevant image-generation disclosure and the editor's interpretation of first-draft restrictions. No false assurances of complete AI policy compliance or figure ownership are made.

Both authors must verify funding/COI statements, author contributions, affiliation/ORCID, data/figure rights, references, ethics, and exclusive submission. The source package and PDFs are prepared for final author review; submission has not been performed and acceptance is not guaranteed. No public repository DOI is invented.
