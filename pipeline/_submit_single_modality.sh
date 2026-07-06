#!/bin/bash
# Launch 8 detached bsub retry-loops (one per modality) so the pending-threshold throttle
# doesn't block the SSH session. Each nohup'd bsub retries every 60s until it lands.
R=/data/salomonis-archive/LabFiles/Nicholas/AML-multimodal/runs/single_modality
for spec in Composition:16000 RNA:48000 ADT:16000 Lipid:16000 Metabolite:16000 GRN:32000 LSC:16000 Cell-comm:64000; do
  M=${spec%%:*}; MEM=${spec##*:}
  nohup bsub -L /bin/bash -q normal -n 4 -W 4:00 -J "sm_$M" -M "$MEM" \
    -R "span[hosts=1] rusage[mem=$MEM]" -o "$R/sm_$M.lsf.out" \
    bash "$R/run_one.sh" "$M" > "$R/sub_$M.log" 2>&1 &
done
echo "launched 8 detached bsub retry-loops"
