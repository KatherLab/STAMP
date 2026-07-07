# STAMP Blast-Enrichment Triage Findings

## Methods
- Analyzed 1020 experiment-slide records from 488 unique sample IDs using 16320 existing heatmap-selected tile JPGs.
- For each complete slide/experiment pair, the script used the existing top_01..top_08 and bottom_01..bottom_08 heatmap tiles, without rereading raw WSIs or rerunning preprocessing.
- Each tile was center-cropped to 512 x 512 px and scored for focus, background, tissue fraction, edge content, and conservative cell-like candidate visibility.
- The blast-enrichment proxy combines positive-class STAMP attention with cellular visibility. It is a triage signal only, not a blast detector.

## Key Findings
- Data coverage was strong: 1020 slide records and 16320 tile records were analyzed; 2 incomplete heatmap folders were skipped.
- The HIGH_BLAST heatmap signal aligned with clinical blast burden: Spearman rho for clinical BLAST_PERCENT vs top enrichment was 0.345 (p=1.02e-15), and HIGH_BLAST AUROC using top enrichment was 0.862.
- The BLAST_PERCENT regression heatmap signal did not align with clinical blast burden: Spearman rho was -0.040, and HIGH_BLAST AUROC from that proxy was 0.424.
- In HIGH_BLAST, high-blast slides had higher top enrichment than low-blast slides (0.301 vs 0.048); confident HIGH_BLAST positives were also much higher than confident negatives (0.321 vs 0.002).
- In BLAST_PERCENT, high-blast slides were not enriched above low-blast slides by this proxy (0.316 vs 0.356), which points away from using the regression heatmaps as the main interim visual triage signal.
- The top 10 BLAST_PERCENT failure slides had very high clinical blast burden (mean GT 87.800%) but low regression predictions (mean 9.436%, mean absolute error 78.364%). Their mean top enrichment proxy was 0.279, not clearly elevated compared with the overall BLAST_PERCENT run.
- The report includes 30 selected overlay contact sheets for technical concerns, BLAST_PERCENT failures, and HIGH_BLAST confidence controls. These overlays are intended for pathologist review and visual sanity checking only.

## Limitations
- The analysis does not classify individual cells and does not identify blasts.
- Green contours are conservative cell-like/foreground candidates, not validated cell detections.
- Clinical BLAST_PERCENT is slide-level metadata; it is not a cell-level or tile-level annotation.
- DeepHeme-style inference is not currently feasible because public DeepHeme assets do not provide a ready pretrained detector/classifier for these STAMP tiles.
- The enrichment proxy can be biased by focus, staining, background, RBC-rich areas, and where STAMP attention lands.

## Recommended Next Evaluations
- Use HIGH_BLAST heatmap tiles, not BLAST_PERCENT regression heatmaps, as the primary interim source for blast-rich candidate regions.
- Ask pathologists to review the selected overlay/contact-sheet examples plus the existing 12-slide pilot deck, with emphasis on whether top-attended regions are cell-rich, blast-rich, or technically misleading.
- Create a small annotation set from high-blast controls, low-blast controls, and BLAST_PERCENT failure slides. Minimum useful labels: interpretable/uninterpretable region, cell-rich yes/no, blast-rich estimate, and artifact/focus notes.
- After pathologist feedback, compare their top-tile blast estimates against the HIGH_BLAST enrichment proxy and clinical BLAST_PERCENT to decide whether to scale annotation.
- If annotations confirm visible blast enrichment, train or calibrate a lightweight cell/region classifier on local annotations. Keep DeepHeme as a reference method unless usable pretrained weights or compatible annotations become available.
- Keep technical QC gates in later analyses: low focus, high background, low candidate density, and top/bottom candidate shifts should be flagged before interpreting model attention.

## Top BLAST_PERCENT Failure Slides
- SAMPLE_396_803_19_B: GT 81.000%, pred 0.121%, abs err 80.879%, top proxy 0.095, technical concern 1
- SAMPLE_396_803_19_A: GT 81.000%, pred 1.230%, abs err 79.770%, top proxy 0.028, technical concern 1
- SAMPLE_408_1411_19_A: GT 91.000%, pred 11.907%, abs err 79.093%, top proxy 0.464, technical concern 1
- SAMPLE_404_1261_19_C: GT 90.000%, pred 11.336%, abs err 78.664%, top proxy 0.225, technical concern 1
- SAMPLE_408_1411_19_C: GT 91.000%, pred 12.751%, abs err 78.249%, top proxy 0.199, technical concern 1
- SAMPLE_404_1261_19_B: GT 90.000%, pred 11.775%, abs err 78.225%, top proxy 0.393, technical concern 1
- SAMPLE_404_1261_19_A: GT 90.000%, pred 11.928%, abs err 78.072%, top proxy 0.499, technical concern 1
- SAMPLE_392_229_19_A: GT 88.000%, pred 10.881%, abs err 77.119%, top proxy 0.187, technical concern 0
- SAMPLE_392_585_19_A: GT 88.000%, pred 11.147%, abs err 76.853%, top proxy 0.373, technical concern 1
- SAMPLE_392_585_19_B: GT 88.000%, pred 11.289%, abs err 76.711%, top proxy 0.326, technical concern 2

Note: all contour overlays are non-diagnostic cell-like candidates, not blast calls.