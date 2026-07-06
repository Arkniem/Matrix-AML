#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
cd /data/salomonis-archive/LabFiles/Nicholas/AML-multimodal/pipeline
exec /usr/local/anaconda3-2020/bin/python stack_meta.py
