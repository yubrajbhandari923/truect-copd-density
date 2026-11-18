# TrueCT COPD — clean Claude Code research workflow

TODO: Explain
## Report
Quarto site in `docs/report/`, published at https://yubrajbhandari923.github.io/truect-copd-density/
(GitHub Pages; the site is public even though the repository is private). Update with
`cd docs/report && quarto publish gh-pages --no-prompt`.

## Data privacy
The repository is public. Real hospital CT (DUHS) series are referred to only by anonymous IDs (`RealCT-Pnn-sNN`);
the mapping to DUHS series names is in `$W/private/realct_id_map.tsv` on the cluster (not in git). Never commit DUHS
series names, patient numbers, or images of real patient CT; anonymise per-case tables before copying them into
`reports/`.
