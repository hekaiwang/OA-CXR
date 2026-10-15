# Score and denominator contract

Higher scores mean more retained anatomy or greater confidence under the
declared adaptation. Regression models clip inference scores to [0,1]. Visual
and report confidence scores retain their original ranking scales and are not
interpreted as retention probabilities.

Every declared query/eligible claim has either a finite successful score or an
explicit failed/null record. Missing rows, duplicates and identity mismatches
are errors. Do not replace a failed score with zero or silently omit its record.
Do not resample sources or stochastic reports based on their outcomes.

Comparisons use a common scored set for exactly the specified methods and report
both nominal and successful denominators. Report bootstrap retains zero-claim
reports in the source/group sampling population. Low-retention positives are
defined by the independent target being below 0.9. Single-class AUROC/AP are
undefined, not zero or one. Bootstrap uncertainty is conditional on fixed models;
it does not measure variation across training seeds.
