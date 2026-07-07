Subject: Interim STAMP blast-enrichment triage findings

Dear all,

While we wait for the pathology review of the initial cell-resolution deck, I ran a conservative interim triage analysis on the existing STAMP heatmap-selected tiles.

In total, this covered 1020 slide/experiment records (16320 tiles) from 488 unique sample IDs. This is not a diagnostic blast detector: it combines STAMP heatmap attention with image-quality and cell-like object visibility metrics.

Main findings:
- The HIGH_BLAST classifier heatmaps showed a meaningful association with clinical blast burden (Spearman rho 0.345; HIGH_BLAST AUROC 0.862).
- The BLAST_PERCENT regression heatmaps did not show useful alignment with clinical blast burden (Spearman rho -0.040; AUROC 0.424).
- The worst BLAST_PERCENT failures had very high clinical blast counts (mean GT 87.800%) but low model predictions (mean 9.436%), suggesting the regression model is not reliably surfacing blast-rich visual regions.

Suggested next step: use the HIGH_BLAST heatmap tiles as the main interim source for candidate blast-rich regions, and ask pathologists to review whether the selected top-attended regions are actually cell-rich/blast-rich or technically misleading. If this is confirmed, we can build a small local annotation set and then train or calibrate a lightweight region/cell classifier.

Best,
Jeff