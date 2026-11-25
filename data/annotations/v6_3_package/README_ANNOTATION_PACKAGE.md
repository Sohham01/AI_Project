# RCI V6.3 — Manual Annotation Package Guide

**Directory:** `data/annotations/v6_3_package/`  
**Purpose:** Prepare strictly blind, non-leaking human annotation batches to unlock genuine multi-human Inter-Annotator Agreement (IAA), authentic Bengali/Odia LID verification, and independent blind evaluation of contextual Issue–Solution linking and thread Resolution detection.

> **Protected Rules Before Opening Any File:**
> 1. **Annotate blindly and independently** across Annotator 1 and Annotator 2 — never share labels before adjudication.
> 2. **Never fabricate or estimate human agreement scores** or fill empty rows with scripts.
> 3. **Keep the frozen `n = 92` Track A test split (`37` threads) untouched** — all Round-2 packages have `0` overlap with it (and `0` overlap with `v1` dev sets where specified).

---

## 1. Inventory & Recommended Annotation Order

| Priority | Package & Purpose | Files Ready for Annotators | Size | Expected Effort | Target Completion |
| :---: | :--- | :--- |---:|---:|---|
| **1** | **`iaa_blind_round2/`** *(Priority #1)*<br>Multi-human IAA (`rci_taxonomy_v2.0` + `ANNOTATION_GUIDELINES_v2.0`) for `sentiment`, `emotions`, `issues`, `response_type`, `is_solution` | • `iaa_blind_round2/round2_blind_80_annotator_1.xlsx` *(or `.csv`)*<br>• `iaa_blind_round2/round2_blind_80_annotator_2.xlsx` *(or `.csv`)*<br>• `iaa_blind_round2/round2_adjudication_template.csv`<br>• `iaa_blind_round2/round2_sampling_manifest.json` | **80 comments**<br>*(29 MBAIndia, 19 IIMSambalpurUG, 23 hi-en, 40 w/ parent)* | ~60–90 min<br>per annotator | True 2-human $\kappa$, Fleiss' $\kappa$, Krippendorff's $\alpha$, MASI $\alpha$ |
| **2** | **`multilingual_bn_or_v2/`** *(Priority #2)*<br>Authentic Bengali/Banglish/Roman-Odia LID & mixed-span benchmark | • `multilingual_bn_or_v2/multilingual_eval_real_v2_candidates.xlsx`<br>• `multilingual_bn_or_v2/multilingual_eval_real_v2_candidates.csv`<br>• `multilingual_bn_or_v2/multilingual_eval_real_v2_candidates.jsonl` | **117 comments**<br>*(14 BN script, 45 Banglish, 23 Roman-Odia, 35 mixed control)* | ~45–60 min | Independent authentic `bn` / `bn-en` / `or-en` LID eval |
| **3** | **`linking_and_resolution_v2/`** *(Priority #3)*<br>Independent unseen-pair Issue–Solution & chronological Resolution evaluation (`0` overlap with `v1` dev) | • `linking_and_resolution_v2/issue_solution_eval_v2_blind.xlsx` *(or `.csv`)*<br>• `linking_and_resolution_v2/resolution_eval_v2_blind.xlsx` *(or `.csv`)*<br>• `linking_and_resolution_v2/v2_protocol_freeze_manifest.json` | **180 pairs** (all 8 issues) +<br>**120 threads** (with timeline digest) | ~90 min (pairs)<br>+ ~120 min (threads) | Blind `v2` independent linking & resolution evaluation |

---

## 2. Exact Column Reference & Allowed Values

### Batch 1 — `iaa_blind_round2/` (`80` Comments × 2 Independent Annotators)
Each annotator fills **only** their own workbook (`annotator_1` or `annotator_2`):
- **Single-label fields (canonical exact spelling):**
  - `sentiment`: `negative` | `neutral` | `positive`
  - `response_type` *(canonical 6-class taxonomy)*: `agreement` | `personal_experience` | `opinion` | `advice` | `solution` | `counterargument`
  - `is_solution`: `true` | `false`
- **Multi-label fields (`|` pipe-delimited, alphabetical preferred):**
  - `emotions`: `confidence`, `curiosity`, `frustration`, `gratitude`, `hope`, `stress` *(or `none` if truly zero affect)*
  - `issues`: `academic_workload`, `administrative_communication`, `campus_infrastructure`, `curriculum_teaching`, `extracurriculars`, `financial_concerns`, `hostel_facilities`, `placements`
- **Governance fields:** `review_status` (`labeled` | `uncertain` | `abstain`), `annotator_id` (`annotator_1` / `annotator_2`), `taxonomy_version` (`rci_taxonomy_v2.0`, prefilled), `guideline_version` (`ANNOTATION_GUIDELINES_v2.0`, prefilled).

### Batch 2 — `multilingual_bn_or_v2/` (`117` Candidates)
*Note:* Columns `language_candidate_detector_hypothesis` and `script_category` are strictly noisy sampling hints (`IndianRedditLID`), **not** ground truth. Fill:
- `gold_language`: `en` | `hi` | `hi-en` | `bn` | `bn-en` | `or` | `or-en` | `und`
- `gold_script`: `latin` | `devanagari` | `bengali` | `odia` | `mixed` | `unknown`
- `gold_is_code_mixed`: `true` | `false`
- `gold_multi_span`: `true` | `false`
- `review_status`: `labeled` | `uncertain` | `abstain`

### Batch 3 — `linking_and_resolution_v2/` (`180` Pairs + `120` Threads)
- **`issue_solution_eval_v2_blind` (`180` pairs):**
  - `relationship_label`: `genuine_solution` | `partial_or_context_dependent` | `not_valid_relationship`
  - `failure_trap_tag` *(optional)*: `contradictory` | `wrong_issue` | `pure_agreement` | `non_actionable` | `insufficient`
  - `review_status`: `labeled` | `uncertain` | `abstain`
- **`resolution_eval_v2_blind` (`120` threads):**
  - `gold_resolution_state`: `reported_success` | `reported_pending` | `proposed_only` | `insufficient_evidence`
  - `gold_verdict_tier`: `Resolved` | `Partially Resolved` | `Unresolved` | `Insufficient Evidence`
  - `gold_has_contradiction`: `true` | `false`
  - `gold_chronological_reversal`: `true` | `false` *(true if later comment reverses an earlier reported outcome)*
  - `gold_evidence_comment_id`: exact `cmt_*` ID from `thread_timeline_digest` supporting your label, or `OP_POST` if resolved/pending in OP text.

---

## 3. Commands to Run After Human Annotation Completes

Once both independent Annotator 1 and Annotator 2 CSVs are saved (without overwriting the original blind templates):

```powershell
# 1. Verify environment
$env:PYTHONPATH = "D:\AI_Project;D:\AI_Project\backend"

# 2. Compute true two-human IAA (Cohen's kappa, Fleiss' kappa, Krippendorff's alpha, MASI alpha)
#    and generate non-destructive disagreement adjudication table
D:\AI_Project\backend\.venv\Scripts\python.exe -c "
import csv, json, pathlib
from ml.evaluation.annotation_reliability import (
    AnnotationSubmission, compute_round_iaa_report, merge_independent_annotations_and_adjudicate
)
def load_subs(path, ann_id):
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        return [
            AnnotationSubmission(
                item_id=r['item_id'], annotator_id=r.get('annotator_id') or ann_id, round_id='round_2',
                sentiment=r['sentiment'], emotions=r['emotions'], issues=r['issues'],
                response=r['response_type'], solution=r['is_solution']
            )
            for r in csv.DictReader(f) if (r.get('review_status') or '').strip() == 'labeled'
        ]
subs_a = load_subs('data/annotations/v6_3_package/iaa_blind_round2/round2_blind_80_annotator_1.csv', 'annotator_1')
subs_b = load_subs('data/annotations/v6_3_package/iaa_blind_round2/round2_blind_80_annotator_2.csv', 'annotator_2')
report = compute_round_iaa_report(subs_a + subs_b, round_id='round_2')
pathlib.Path('ml/artifacts/v6_3_experiments/annotation_audit/round2_iaa_results.json').write_text(
    json.dumps(report, indent=2), encoding='utf-8'
)
merged = merge_independent_annotations_and_adjudicate(subs_a, subs_b)
print(json.dumps(report['summary'], indent=2))
"

# 3. Run existing automated annotation & blind protocol tests
D:\AI_Project\backend\.venv\Scripts\python.exe -m pytest backend/tests/test_annotation_reliability.py -v
```
