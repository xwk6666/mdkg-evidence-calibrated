# Frozen metric conventions

Constructed-pair Table4/Fig4 retain the historical stable-sort rank AP convention
implemented by the included original evaluate_image_text_consistency.py. Exact tied
scores preserve file order. Grouping tied thresholds instead produces a slightly
different AP, so that grouped implementation is not silently substituted for the
historical table. AUROC uses mid-ranks for ties in all replay paths.

Human300, MASH and fixed250 NLI use grouped-score Average Precision matching their
frozen evaluation convention. The replay driver reports the convention explicitly.
No metric choice, model, score or threshold was optimized on Gold. All original
records remain unchanged. Main Table8 uses raw single/multi-template NLI at archived
development-selected cutoffs; P(r)-tier quality is a separate fixed250 calculation.
