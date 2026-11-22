# Archived Experimental Expansion — `real_reddit_labeled_v2` (Blocked by Human IAA Gate)

**Status:** `ARCHIVED / NOT USED IN PRODUCTION`

This directory preserves the Track B uncertainty-sampled expansion (`707` real comments: `459` V1 + `248` newly curated real Reddit comments) strictly for research provenance and future adjudication.

- **Why it is excluded from production:** Before promoting V2, a genuine 50-item blind Human IAA audit was completed in the local Annotation Studio (`docs/V5_HUMAN_IAA_REPORT.md`). All `5 / 5` tasks fell below the pre-registered agreement gate ($\kappa \ge 0.70 / 0.65$, MASI $\alpha \ge 0.65$) due to six taxonomy boundary ambiguities.
- **Canonical Production Ground Truth:** Production models (`v6.1.0`) train and evaluate exclusively on the untouched Track A dataset in `data/annotations/real_reddit_labeled_v1/` (`459` real comments across `200` distinct threads).
- **Unlock Requirement:** Re-annotate the 30-item blind calibration batch (`iaa_calibration_30.csv`) using the six mandatory tie-breakers in `docs/ANNOTATION_GUIDELINES.md` Section 8 and verify all 5 tasks clear their IAA threshold gates.
