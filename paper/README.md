# Revised research paper

`main.tex` is the updated version of the supplied IEEE LaTeX draft. Upload the contents of the Overleaf ZIP and set **main.tex** as the main document. Use **XeLaTeX** to match the locally checked engine family; Overleaf will run BibTeX automatically. The source also uses ordinary pdfLaTeX-compatible packages, but a separate pdfLaTeX build was not run here.

Required build files are included:

- `main.tex`, `supplementary.tex`, `references.bib`
- `tables/*.tex`
- `figures/*.pdf`

The revised manuscript uses verified project records for the three validation runs and the selected fold-1 test result. It includes the author information supplied in the conversation. It does not invent missing experiments or treat validation differences as test improvements.

Read `evidence_audit.md` for claim-by-claim provenance and literature corrections. Read `missing_or_unverified.md` for the remaining research gaps. Machine-readable recomputation results are in `audit/verification.json`; original-record hashes are in `audit/evidence_manifest.csv`. Numeric tables are included as CSV as well as LaTeX. `evidence/` contains copies of the original prediction CSV, aggregate results, training histories/configurations, environment record, and dated validation selection review. `audit/original_pasted_draft.txt` preserves the starting text for comparison.

`audit_and_figures.py` rebuilds the evidence checks and graphics from the original repository assets. Those large audio and model files are not distributed with the Overleaf package. Compilation itself requires none of them, no Python, and no access to the project's training machine.

The supplementary material is included as appendices. The paper may need shortening for a specific venue's page limit, which has not been supplied.

Local validation: the seven-page `main.pdf` was compiled with Tectonic 0.17.0 (XeTeX and BibTeX), and all seven pages were visually inspected. There are no unresolved citations/cross-references, missing figures, or overfull boxes. Standard class/font-initialization and underfull-line warnings remain in the saved build log; the rendered text, equations, figures, and tables were checked. See `audit/document_checks.json` and the build logs for details.
