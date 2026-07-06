#!/usr/bin/env python3
"""End-to-end validation of the scRNA composition keystone on a 10x CITE-seq sample
(Grimes AML-CITE-Seq: filtered_feature_bc_matrix.h5).

Loads the GENE-EXPRESSION features only (drops Antibody-Capture/ADT), cosine-assigns every
cell to the 89 bone-marrow reference states, and reports the cell-state COMPOSITION + alignment
quality. A working run here closes the open wrinkle: the panel's `composition` is now derivable
from a real gene-level scRNA query (the KINNEX h5ads were junction-level → 0 markers).

Run on an LSF COMPUTE node (not the head node):
    bsub -q normal -K -M 8000 -R "rusage[mem=8000]" \
      /usr/local/anaconda3-2020/bin/python _scrna_cite_validate.py [sample_dir_or_h5 ...]
"""
from __future__ import annotations
import os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from amlmm import scrna

DEFAULT = [
    "/data/salomonis-archive/LabFiles/Nathan/Collaborators/Grimes/AML-CITE-Seq/AML-7",
    "/data/salomonis-archive/LabFiles/Nathan/Collaborators/Grimes/AML-CITE-Seq/BF71-CD34",
]


def _resolve(p):
    return os.path.join(p, "filtered_feature_bc_matrix.h5") if os.path.isdir(p) else p


def load_gex(path):
    """(AnnData with Gene-Expression features only, feature_type Counter, loader-used).
    Prefer scanpy.read_10x_h5; fall back to parsing the 10x-v3 HDF5 layout with h5py."""
    try:
        import scanpy as sc
        a = sc.read_10x_h5(path, gex_only=False)
        ft = collections.Counter(str(x) for x in a.var.get("feature_types", []))
        if "feature_types" in a.var:
            a = a[:, a.var["feature_types"].astype(str) == "Gene Expression"].copy()
        a.var_names_make_unique()
        return a, ft, "scanpy"
    except Exception as e:
        sys.stderr.write(f"[scanpy unavailable/failed: {type(e).__name__}: {e}; using h5py]\n")
    import h5py, anndata as ad, pandas as pd
    from scipy import sparse
    with h5py.File(path, "r") as f:
        m = f["matrix"]
        bc = [x.decode() for x in m["barcodes"][:]]
        name = np.array([x.decode() for x in m["features"]["name"][:]])
        ftype = np.array([x.decode() for x in m["features"]["feature_type"][:]])
        shp = m["shape"][:]
        X = sparse.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]),
                              shape=(int(shp[0]), int(shp[1]))).T.tocsr()   # cells x features
    gex = ftype == "Gene Expression"
    a = ad.AnnData(X=X[:, gex], obs=pd.DataFrame(index=bc),
                   var=pd.DataFrame(index=name[gex]))
    a.var_names_make_unique()
    return a, collections.Counter(ftype.tolist()), "h5py"


def main():
    queries = [_resolve(p) for p in (sys.argv[1:] or DEFAULT)]
    ref = scrna.load_reference()
    print(f"reference: {ref.shape[0]} markers x {ref.shape[1]} populations")
    for q in queries:
        print(f"\n==== {q} ====")
        if not os.path.exists(q):
            print("  [missing]"); continue
        a, ftc, how = load_gex(q)
        print(f"  loaded via {how}; feature_types={dict(ftc)}")
        print(f"  GEX matrix: {a.shape} (cells x genes); first genes: {list(a.var_names[:10])}")
        try:
            res = scrna.composition_from_query(a)
        except ValueError as e:
            print(f"  CANNOT ASSIGN: {e}"); continue
        comp = res["composition"]
        print(f"  shared markers: {res['n_shared_markers']} | mean cosine: {res['mean_cosine']:.3f}"
              f" | n_cells: {res['n_cells']} | states present: {res['n_states_present']}/89")
        print("  top cell-states (composition):")
        for s, v in comp.sort_values(ascending=False).head(14).items():
            print(f"    {s:<30} {v:.3f}")
    print("\nDONE — composition Series is a drop-in for the panel.")


if __name__ == "__main__":
    main()
