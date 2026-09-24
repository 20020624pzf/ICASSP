# Scientific scope

Whisper-small with frozen rank-four R0 LoRA (884,736 parameters) generates a
shared greedy plus native beam-5 bank with normalized deduplication, at most six
texts. Seeds 101/202/303 vary the ASR checkpoint; they are not independent selector
fits. Known-language transcription, bfloat16, 444 new tokens, deterministic
decoding and `length_penalty=1.0` are preserved in the full configuration files.

Conf's eight features include clean-audio conditional probabilities, lengths,
repetition/cap indicators, consensus and LCS-local clean confidence. Global adds
the beam-relative clean-minus-permuted mean; Local also aggregates at text-LCS
disagreements (not acoustic time alignment). Text adds the difference in frozen
XLM-R mean masked-token PLL from beam; Text+Global includes both. These are
alternative fitted selectors, not an ensemble. Only Global/Local/Text+Global
require the 20-frame block-permuted acoustic pass.

One seed101 TRAIN-FIT fit serves all three checkpoints. Calibration has 1,281
recordings: 1,016 FIT and 265 diagnostic-only CHECK. The preparation script fixes
language-specific text-hash groups and FIT/CHECK membership. Weighted RMS is
computed on FIT only; ridge has no intercept and lambda 0.01. Strict positive
predicted gain replaces beam, and first-argmax resolves ties. References supervise
TRAIN targets and final scoring, never inference selection. Recording IDs fix
the permutation but are not predictor features.

Original Local, fixed checkpoint extensions, MWER, frozen-text controls, and later
parameterization/fit sensitivity diagnostics retain their real post-exposure
timeline and separately reported statistical families. MWER changes both fitting
objective and candidate weighting. Half-FIT diagnostics are sensitivity analyses,
not confidence intervals over independent fitted models.

Metric: WER-v3; 100*(S+D+I)/N within language, macro average over five languages,
then mean and sample SD over the three fixed checkpoints. Reported differences
are percentage points, computed before display rounding. Full references are
retained despite 30-second input windows. Tables preserve original Local/MWER
results and DEV reversals.

Cost: fixed seed101, 200 test recordings, five systems and three paired warm
repetitions = 3,000 rows. Pooled ratios are ratios of summed warm times under
observed shared load, not universal overhead or production SLA. XLM-R requires
additional model storage/memory. CPU cache timing excludes candidate acquisition
and acoustic/MLM inference.
