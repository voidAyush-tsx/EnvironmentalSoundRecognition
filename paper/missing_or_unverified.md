# Remaining missing or unverified items

The project now has enough local evidence for the specific single-fold results in `main.tex`. The following items cannot honestly be filled from the available records.

| Item | Current status | What would be needed |
| --- | --- | --- |
| Ten-fold mean, standard deviation, and fold-wise robustness | Not measured in the recovered final comparison | Ten independent official test-fold runs with validation-only selection and preserved predictions |
| Variation across random seeds | One recorded seed (67) in the comparison | Prespecified repeated-seed runs |
| Held-out improvement of `block6` over `head` or `fc1` | Only validation comparisons exist; only `block6` was tested | A properly designed additional evaluation, acknowledging the already observed fold-1 results |
| Full fine-tuning, blocks 5-6, or blocks 4-6 ablations | No completed matching runs recovered | New controlled experiments; candidate rows were removed instead of populated |
| Strict source-independent generalization | Four source IDs cross training/test; one crosses training/validation | A separately labeled source-disjoint sensitivity protocol and new training/evaluation, rather than silently changing the historical splits |
| Statistical significance / confidence intervals accounting for sources | Not established | Appropriate repeated/group-aware evidence and a prespecified analysis |
| Inference latency, real-time factor, energy use | Not measured | Device-specific, reproducible measurement protocols |
| Repeated runtime/memory reliability | One recorded run per configuration | Repeated measurements under specified machine conditions |
| Training-mode versus evaluation-mode generalization gap | Online training and evaluation-mode validation differ | Fixed-checkpoint evaluation of both partitions under matching conditions |
| Calibration, external-dataset, field-device, long-recording, out-of-distribution robustness | Not measured | Dedicated evaluations |
| Overlap between AudioSet pretraining and UrbanSound8K recordings | Not audited | Cross-dataset recording/duplicate analysis |
| Exact project Git revision at the time of training | Not included in run configs | Contemporaneous source snapshot/hash; the inspected current revision is supplied but not substituted for it |
| Independent replay of predictions from the selected model | Not performed for this writing task | Explicit fresh inference if required; current verification recomputes saved predictions and verifies checkpoint identity/reconstruction |
| Independent reproduction of historical training and timings | Not performed | Full reruns with preserved provenance |
| Bit-identical recomputation of test loss from original logits | Original logits not retained | Original logits or inference replay; 0.3800 is explicitly a recorded loss |
| Submission venue, page limit, and venue-specific formatting | Not supplied | Intended venue requirements; current output retains IEEE conference format |

Author names, department, institution spelling, location, and Ayush Ranjan's email were supplied directly by the user. The user confirmed that Sandeep Gupta shares the affiliation and requested omission of his email. These details are no longer placeholders. The institution's supplied spelling is preserved rather than independently changed.

No claim of a new architecture, state of the art, statistically proven superiority, ten-fold validation, or strict source independence should be added on the strength of the current evidence. Results from an ESC-50 experiment have not been used to fill UrbanSound8K gaps.
