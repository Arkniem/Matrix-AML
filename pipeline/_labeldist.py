from amlmm.context import build_context, Config
from amlmm import discovery as D, targets
ctx = build_context(Config(run_id="_ldist"))
for f in ["subtype", "ELN_risk", "is_pediatric", "disease_category"]:
    lab = D.labels_for_field(ctx, f).dropna()
    vc = lab.value_counts()
    usable = targets.usable_classes(lab, 8)
    print(f, "| labeled", len(lab), "| usable(min8)", usable)
    print("   dist:", dict(vc.head(8)))
