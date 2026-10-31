# Source corpora

This directory is empty on purpose. Run `python3 get_data.py` to populate it:

- `popquorn_politeness_raw.csv` — POPQUORN politeness ratings (Pei & Jurgens
  2023), from the authors' repository. Verified against the SHA-256 of the
  exact file the paper's results were computed from.
- `mhsc.parquet` — Measuring Hate Speech Corpus (Kennedy et al. 2020), from
  HuggingFace. Checked by shape (135,556 rows, 7,912 annotators) rather than
  checksum, because the parquet is re-encoded upstream without the data
  changing.

Both remain under their authors' licences and are not redistributed here.
