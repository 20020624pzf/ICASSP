# Data and model access

This repository publishes aggregate WER/S/D/I/N tables, anonymous integer client
aggregates, and anonymous timing rows. It does not publish source audio, corpus
reference or candidate transcripts, original client/recording IDs, private caches,
credentials, or model weights. Integer client ordinals preserve pairing for the
existing bootstrap without an identity lookup. Timing rows use local ordinals
0--199 rather than original recording IDs.

| Asset | Version and access route |
|---|---|
| Whisper-small | `openai/whisper-small`, revision `973afd24965f72e36ca33b3055d56a652f456b4d`, https://huggingface.co/openai/whisper-small |
| XLM-R base | `FacebookAI/xlm-roberta-base`, model/tokenizer revision `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`, https://huggingface.co/FacebookAI/xlm-roberta-base |
| FLEURS | Official ar_eg/fa_ir/pl_pl/tr_tr/it_it TSV releases, CC BY 4.0; https://huggingface.co/datasets/google/fleurs and https://github.com/google-research-datasets/fleurs |
| CV27 | `cv-corpus-27.0-2026-09-11`, ar/fa/pl/tr/it; obtain under applicable terms at https://mozilladatacollective.com/datasets |
| R0 LoRA adapters | Seeds 101/202/303, exact SHA-256 records in `VERSIONS_AND_ASSETS.json`; weights are not hosted here |

The five recorded CV27 dataset records specify CC0-1.0 and access restrictions
against speaker reidentification and dataset re-hosting/re-sharing. Obtain them
from the authorized distributor; repository publication does not extend their
terms. Preserve original dataset attribution. Historical FLEURS manifests record
TSV hashes, not an upstream repository commit; no missing revision is invented.

All source content hashes and recorded counts are in
`reproduction/config/VERSIONS_AND_ASSETS.json`. Some historical provenance entries
contain inert server paths to identify original assets; local input paths must be
provided through the portable configuration.

FLEURS TEST: 3,665 recordings including 22 longer than 30 s. The existing study
uses 30-s ASR inputs and complete references. CV27 TEST: 58,514 recordings. These
are the manuscript's original protocols, not a claim of completed long-audio
reevaluation. The separate unfinished full-audio DEV study is not released as a
result of this paper.
