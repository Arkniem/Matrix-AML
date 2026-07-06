# MATRIX-AML retrospective clinical validation

_Retrospective ASSOCIATIONS on sparse, partially-overlapping clinical labels — NOT trained predictors. n and an underpowered flag (<20) accompany every test; a null is a legitimate finding. ELN-expected is a single-driver proxy for a multi-factor clinical schema. CIRCULARITY: ELN risk is itself defined from driver genetics, so the anchored-driver ELN concordance + the adverse-driver->ELN test SHARE logic with the label — they validate the engine's driver EXTRACTION + anchor selection (a correctness/data-integrity check), NOT a novel prediction. The p-LSC->ELN test is the more INDEPENDENT signal (an RNA-derived stemness phenotype tracking genetics-defined risk), as is the driver->clinical-response test._

## Coverage (the binding constraint)
- samples: 387; anchored driver: 153; ELN: 70; clinical response: 52; survival: 19; confident LSC: 247

## Tests
- **Anchored-driver ELN-expected vs clinician ELN (3-class concordance):** agreement 0.744 (n=39, perm p=5e-05) _[shares-logic (extraction/anchor correctness check)]_
- **Adverse anchored driver -> ELN Adverse:** OR=144.0 (n=39, Fisher p=0.0, 2x2=[[12, 1], [2, 24]]) _[shares-logic (extraction/anchor correctness check)]_
- **Favorable anchored driver -> clinical responder:** OR=0.037 (n=48, Fisher p=2e-05, 2x2=[[2, 17], [22, 7]]) _[independent (genetics vs treatment response)]_
- **Primitive-LSC (p-LSC) -> ELN Adverse:** OR=2.625 (n=44, Fisher p=0.44505, 2x2=[[15, 20], [2, 7]]) _[independent (RNA-derived stemness vs genetic risk)]_
- **Median survival (months) by expected risk** (descriptive, tiny n): Intermediate: 773.0 (n=2); Adverse: 92.0 (n=6)

### Anchored-driver distribution (cohort)
NPM1=43, Inv16=19, FLT3=14, TET2=12, TP53=10, Complex=8, KMT2Ar=7, DEL(7)=7, Trisomy8=6, SF3B1=6, t(8;21)=4, SRSF2=3, DNMT3A=2, CEBPA=2, IDH2=2, WT1=2, ASXL1=2, IDH1=1, NRAS=1, CSF3R=1, DEL(5)=1