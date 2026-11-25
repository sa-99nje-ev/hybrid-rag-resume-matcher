# Independent annotation guide

The local `data/evaluation/djinni_random_annotation.csv` contains 50 real pairs: five hybrid-retrieved candidates for each of ten randomly selected sampled jobs. All relevance, annotator and notes fields start blank. The file contains candidate profile text; keep it local and follow applicable access and retention rules.

Review each job and candidate independently. Choose and document a grading rubric before annotation. Record evidence and uncertainty; absence of evidence does not prove absence of a skill. Do not use an LLM or retrieval score as the annotator. If possible, obtain two independent reviews, retain disagreements, and document how they were resolved. Reserve separate job groups for validation and testing; audit document, exact-text and near-duplicate overlap across all partitions.

These candidates were selected by one hybrid system, so judgments are pooled and incomplete. A blank pair is unjudged, not irrelevant. Broaden the pool across methods and depth before comparing their relevance; measure and report judgment coverage. Do not use the source-ordered benchmark as an independent split: 518 source IDs overlap the current sample. Avoid deriving positive or negative labels from source order, category, embedding similarity or generated output.
