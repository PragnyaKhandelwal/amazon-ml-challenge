# Business Entity Resolution — Amazon ML Challenge 2026

For every Source-1 business record, find all Source-2 / Source-3 records that refer to
the same real-world business (US, India, and France in test). Metric: per-entity F0.5,
macro-averaged, singletons included.

**Status:** pipeline validated on a 10% development subset of train —
**0.9885 macro F0.5** on held-out Source-1 entities. Full-scale run pending (see
[docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md)).

## Approach

```
raw TSVs -> 1 normalise -> 2 block -> 3 prune + features -> 4 LightGBM -> 5 decide -> TSVs
```

1. **Normalise** (`prepare.py`, `normalize.py`, `translit.py`): ASCII folding, OCR repair,
   legal-form canonicalisation, DBA / website / domain-name handling, address abbreviation
   and ordinal canonicalisation, consonant-skeleton keys, and a native-script → Latin token
   table learned from training pairs for Indic-script names.
2. **Block** (`run_blocking.py`, `blocking.py`): weighted rare-key join within each
   country label (name tokens, skeletons, token pairs, name × rare address word,
   house number × rare address word, address-word pairs, squashed names). Keys shared by
   more than 300 Source-1 records are dropped; each target keeps its top-10 Source-1
   candidates by summed IDF.
3. **Features** (`pairs.py`, `features.py`): candidates below 25% of the target's best
   blocking score are pruned; ~140 country-agnostic features (rapidfuzz similarities,
   IDF-weighted overlaps, number agreement, name/address frequency, and context features
   comparing each pair with its competitors).
4. **Matcher** (`train.py`, `model.py`): LightGBM binary classifier; Source-1 entities are
   split into 5 hash folds and fold 0 is held out from every fitted component.
5. **Decide** (`evaluate.py`, `predict.py`): each target is assigned to its single best
   Source-1 entity if the probability clears a threshold tuned for macro F0.5 on fold 0.

No external data, APIs or pretrained models are used.

## Reproduce

Requires Python 3.11 and the challenge data (not included in this repo).

```bash
pip install -r requirements.txt
export ER_DATA_DIR=/path/to/student_resource/dataset   # contains train/ and test/
export ER_WORK_DIR=/path/to/work                       # ~10 GB of intermediates
export ER_OUT_DIR=/path/to/output
export ER_WORKERS=8

cd src
python prepare.py train test          # normalise (+ learn transliteration table)
python run_blocking.py train 10 300   # candidates + recall report
python run_blocking.py test 10 300
python train.py all 0.25 1500         # features, model, validation F0.5 -> work/stage1.json
python predict.py all                 # output/matching_results.tsv + candidate_pairs.tsv, then validator
```

Full scale needs ~32–64 GB RAM. `experiments/` holds the development-subset scripts
(hard-coded local paths).

## Layout

| Path | Contents |
| --- | --- |
| `src/` | pipeline source |
| `experiments/` | dev-subset blocking, training, pruning and error-analysis scripts |
| `docs/PROJECT_STATUS.md` | findings, results, engineering notes, cloud runbook |
