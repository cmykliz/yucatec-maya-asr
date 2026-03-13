# Yucatec Maya ASR — Fine-tuning MMS on a Low-Resource Language

A speech recognition pipeline for Yucatec Maya (ISO 639-3: `yua`), fine-tuned from Meta's
[Massively Multilingual Speech (MMS)](https://huggingface.co/facebook/mms-300m) model using the
[DoReCo corpus](https://doreco.huma-num.fr/). Built as a portfolio project to demonstrate
end-to-end NLP/speech engineering on a genuinely low-resource language, and as a contribution
toward tools that support the work of indigenous language preservation organisations such as
[ALMG (Academia de las Lenguas Mayas de Guatemala)](https://www.almg.org.gt/).

---

## Language Background

**Yucatec Maya** is a Mayan language spoken primarily in the Yucatán Peninsula of Mexico and
parts of Belize, with approximately 800,000 speakers. It has a rich morphological system,
including vowel length distinctions, tonal contrasts, and ejective consonants — features that
make ASR particularly challenging. Despite its relatively large speaker population for an
indigenous language, it remains severely under-resourced in NLP.

---

## Dataset

**Source:** [DoReCo 2.0](https://doreco.huma-num.fr/) — Documentary Reference Corpus  
**Licence:** CC BY-NC-ND 4.0  
**Recordings:** 10 WAV files, 6 speakers (ages 23–45, 5F/1M)  
**Genres:** Personal narrative, procedural, traditional narrative

| Split | Files | Utterances | Duration |
|-------|-------|-----------|----------|
| Train | 8     | ~694      | ~0.89h   |
| Test  | 2     | ~164      | ~0.22h   |

**Note on data volume:** ~1.1 hours of transcribed speech is very small for ASR training.
Most successful fine-tunes use 10–100+ hours.

### Data Processing: `process_doreco.py`

Raw DoReCo data consists of WAV recordings paired with ELAN `.eaf` annotation files. The script:

1. Parses `ref@` tier utterance boundaries (millisecond timestamps)
2. Filters out pause markers (`<p:>`) and utterances shorter than 1.5 seconds
3. Slices WAV files into utterance-level clips using `pydub`, resampled to 16kHz mono
4. Outputs a HuggingFace `audiofolder`-compatible dataset with full metadata

---

## Model

**Base model:** [`facebook/mms-300m`](https://huggingface.co/facebook/mms-300m)  
**Architecture:** Wav2Vec2ForCTC (300M parameters)  
**Approach:** CTC fine-tuning with a freshly initialised language model head (44-token
Yucatec Maya character vocabulary)

The feature encoder is frozen during training — only the transformer layers and LM head
are updated. Standard practice for small datasets to reduce overfitting risk.

---

## Results

| | WER |
|---|---|
| Baseline (zero-shot MMS) | 1.056 |
| Fine-tuned (30 epochs) | **0.991** |

A WER of 0.991 means the model still gets ~99% of words wrong. This is an honest result
that deserves honest interpretation:

**Why the WER is high:**
- ~1.1 hours of training data is insufficient for robust ASR (typical threshold: 10–100h)
- Yucatec Maya orthography includes phonologically complex characters (ejectives `ʼ`, glottal
  stop `ʔ`, vowel length) that require more examples to learn reliably
- The model did learn — training loss dropped from ~1900 to ~390 over 30 epochs, and WER
  improved from 1.056 to 0.991 — but data volume is the binding constraint

**What this project demonstrates:**
The pipeline is correct and complete. Given sufficient data (e.g. from ALMG archives or
future DoReCo releases), the same code would be expected to produce meaningful WER
improvements. The value here is in the infrastructure, not the current metric.

---

## Training Details

| Hyperparameter | Value |
|---|---|
| Base model | facebook/mms-300m |
| Epochs | 30 |
| Batch size | 2 (effective 4 with gradient accumulation) |
| Learning rate | 1e-5 (linear warmup 200 steps) |
| Max grad norm | 0.1 |
| CTC loss reduction | mean |
| Hardware | Apple M2 Max (MPS) |
| Training time | ~77 minutes |

---

## Repository Structure

```
yucatec-maya-asr/
├── process_doreco.py      # EAF annotation parser → HuggingFace dataset
├── finetune_mms.py        # MMS fine-tuning script (CTC, HuggingFace Trainer)
├── environment.yaml       # Conda environment specification
└── README.md
```

---

## Reproducing This Work

### 1. Set up environment
```bash
conda env create -f environment.yaml
conda activate mayan-asr
```

### 2. Download the DoReCo Yucatec Maya corpus
Visit https://doreco.huma-num.fr/ and download the Yucatec Maya (`yuca1254`) package.
Place audio files and annotation files in separate directories.

### 3. Process the dataset
```bash
python3 process_doreco.py \
    --audio_dir doreco_yuca1254_audiofiles_v2.0 \
    --eaf_dir doreco_yuca1254_core_v2.0 \
    --output_dir ./doreco_yuca1254_hf \
    --metadata_csv doreco_yuca1254_audiofiles_v2.0/doreco_yuca1254_metadata.csv
```

### 4. Fine-tune
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 finetune_mms.py \
    --data_dir ./doreco_yuca1254_hf \
    --output_dir ./mms-300m-yucatec-maya \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-5
```

> **Apple Silicon note:** `PYTORCH_ENABLE_MPS_FALLBACK=1` is required because CTC loss
> is not yet implemented for MPS and falls back to CPU.

---

## Limitations and Ethical Considerations

- **Data volume:** Results are limited by the small size of the DoReCo corpus. This is a
  fundamental constraint of low-resource language work, not a modelling failure.
- **Speaker diversity:** 6 speakers from a single corpus may not represent the full
  dialectal variation of Yucatec Maya.
- **Licence:** The DoReCo corpus is CC BY-NC-ND 4.0 — non-commercial use only. The
  fine-tuned model weights should be treated accordingly.
- **Community consent:** This project uses data collected by linguists working with Maya
  communities. Any deployment of ASR tools for Yucatec Maya should be done in partnership
  with those communities and organisations like ALMG.

---

## Next Steps

- [ ] Obtain access to larger Yucatec Maya audio archives (outreach to ALMG ongoing)
- [ ] Extend pipeline to K'iche' Maya using the same architecture
- [ ] Publish fine-tuned model to HuggingFace Hub with model card
- [ ] Error analysis: which phonemes/morphemes does the model struggle with most?
- [ ] Explore data augmentation (speed perturbation, noise injection) to compensate for
      small dataset size

---

## Acknowledgements

DoReCo corpus compiled by Seifart et al. Yucatec Maya recordings courtesy of the DoReCo
project contributors. MMS model by Meta AI Research.

---

## Citation

If you use this pipeline, please cite the DoReCo corpus:

```
Seifart, F. et al. (2022). DoReCo: Documentary Reference Corpus (Version 2.0).
https://doreco.huma-num.fr/
```
