# AML multimodal pipeline run: smoke_local

- samples: 387 | pseudobulks: 12255 | modalities: RNA, GRN, Metabolite, Lipid, ADT, cell-communication
- step `feasibility` [ok, 0.06s]: n_targets=9, trainable=['subtype', 'disease_category', 'ELN_risk', 'sex', 'age'], marginal=[], not_trainable=['WHO_classification', 'FAB', 'is_pediatric', 'overall_survival']
- step `assemble_features` [ok, 0.0s]: target=subtype, blocks=['composition'], n_samples=97, n_features=90, n_classes=5, class_counts={'NPM1': 43, 'Inv16': 18, 'FLT3': 14, 'TET2': 12, 'TP53': 10}
- step `classify` [ok, 22.84s]: target=subtype, strategy=donor_kfold, models=['rf', 'logreg'], balanced_accuracy=0.6456, macro_f1=0.6248, n_samples=97
- step `cluster_explore` [ok, 0.03s]: n_programs=16, n_pseudobulks_assigned=8564, programs_dataset_dominated_gt80pct=1, example_batch_dominated=['P4'], max_dataset_fraction_median=0.378

**Gate:** ACCEPT — balanced_acc 0.646 above chance (n=97) (action=pass)