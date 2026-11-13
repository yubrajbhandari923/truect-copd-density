"""Rebuild the summaries of a validation.json from its per-subject metrics (no recomputation).
usage: resummarize.py validation.json   (rewrites in place; used once to fix the material_dice summary of job 204991)"""
import json, sys
sys.argv, path = sys.argv[:1], sys.argv[1]
from baselines import summary
v = json.load(open(path))
v["summary"] = {k: summary(v["per_subject"], m) for k, m in v["chosen_variant"].items()}
v["all_variants_summary"] = {m: summary(v["per_subject"], m) for m in next(iter(v["per_subject"].values()))}
v["provenance"]["summary_rebuilt_by"] = "experiments/e01/resummarize.py (per-subject metrics unchanged)"
json.dump(v, open(path, "w"), indent=1)
