# Evidence-calibrated multimodal disaster knowledge graphs

Local v1.0.0 publication draft. No public repository, release or archive DOI exists
for this draft. Do not upload it without the author's separate approval.
Suggested repository: mdkg-evidence-calibrated.

## Research problem and framework

Noisy image-text pairs and unsupported generated relations complicate knowledge
graph construction from disaster social media. The framework separates pairing,
candidate structure and source-text relation support instead of treating all three
as one confidence score.

CLIP with same-disaster-type/different-family Semantic-HN learns sample-level
observed-versus-constructed pairing. Platt calibration produces q_i^m; it is
recorded on ALIGNED_WITH, not an independently verified event-truth probability.
Closed-schema LLM extraction proposes Disaster, Location, Infrastructure and
Casualty entities and OCCURS_IN/AFFECTS/CAUSES relations. Optional type-compatible
proposal expansion adds recall-first proposals, not factual support.
Structural validation precedes full-post multi-template NLI. StandardScaler and
L2 logistic calibration estimate P(r), followed by frozen Reliable/Review/Reject
thresholds. q is not fused with P(r) and does not gate relation admission.
Neo4j organizes admitted semantic relations and separately stored pairing evidence
for event queries and provenance tracing. Reliable is system eligibility, not
external factual certification; Review remains pending human review.

## Contents

src/ contains 19 selected original method modules. scripts/frozen_metrics.py and
scripts/reproduce.py are byte-identical to the audited core replay modules.
scripts/reproduce_public.py is an I/O-only adapter for the separate data archive:
it joins Human300 by the existing annotation_id and does not invent split IDs.
schema/ and prompts/ are current source specifications/templates. They are not
claimed to be a complete immutable historical API-call record.

## Installation and environment

Frozen replay needs Python 3.10 or newer and only the standard library; no model,
API key, network or database is used. No package installation is needed for replay.
The checked staging runtime is Python 3.12.14, not the reconstructed training runtime.
See ENVIRONMENT.md. requirements-inference.in is an unpinned dependency inventory
for separately authorized future inference, not a historical environment lock.
Do not run training, calibration, LLM or database entrypoints as part of replay.

## Original data access

Follow DATA_SOURCES.md and each upstream provider's terms to obtain CrisisMMD v2.0
and MASH. This repository contains no raw posts, media, weights, original split
membership lists or endpoint Gold. Data fetching/training is not automated here.

## Frozen numerical reproduction

Extract the separately reviewed data archive as a sibling zenodo_dataset_release/
directory, or give its actual path. Run from this software directory:

```sh
python -B scripts/reproduce_public.py --data-root ../zenodo_dataset_release --target core
python -B scripts/reproduce_public.py --data-root ../zenodo_dataset_release --target all
```

Individual targets include table4/table6/table8/table9/table10/table11 and
fig6/fig9. The adapter verifies both SHA256 manifests before reading data.
METRIC_CONVENTIONS.md preserves Table4's historical stable-rank AP; Human300/MASH
use grouped-score AP. No Gold threshold is selected and no calibrator is refitted.

## Reproduction scope

**Frozen-output reproduction, not full end-to-end reproduction.** The public
projection keeps all rows, labels, frozen scores and ordering but omits linkable
upstream IDs. Original split membership is not publicly revalidated.
Raw entity/endpoint matching is not rerun. Some targets expose archived summary
numbers only. See PAPER_EVIDENCE_MAP.md, REPRODUCIBILITY.md and known_limitations.md
for Fig6 CI, Fig9 snapshot and Table9 denominator boundaries. Historical200 and
prospective reused200 are separate protocols, not two independent input samples.

## Licenses and attribution

MIT applies to authorized original software and its documentation/specifications
in this repository, as bounded by LICENSE_SCOPE.md. The separate archive's original
derived labels/numerical outputs are CC BY 4.0. Neither license covers upstream
posts/media, third-party dependencies, model weights or the manuscript.
Third-party dependencies are imported, not vendored; see THIRD_PARTY_NOTICES.md.

## How to cite

Use CITATION.cff for the five authors in manuscript order and software version.
Paper: An Evidence-Calibrated Framework for Constructing Multimodal Disaster
Knowledge Graphs from Social Media; manuscript, not a claimed published EAAI article.
Code URL: [REAL_GITHUB_REPOSITORY_URL_PENDING]
Dataset URL/DOI: [REAL_ZENODO_RECORD_URL_AND_DOI_PENDING]
Replace these only after actual creation, publication and verification. No ORCID,
publication date or article DOI has been invented.
