# Data availability

This repository contains the SwarmWorld simulator, interfaces, configurations, tests,
and documentation. It intentionally does not contain raw or derived paper data.

The complete paper dataset is available separately from Hugging Face:

<https://huggingface.co/datasets/lamm-mit/swarmworld-data>

The approximately 8.4 GB dataset contains 60 analyzed episodes across nine study
invocations, including authoritative compressed event traces, isolated-search member
traces, pre-run manifests, episode summaries, seed-level tables, derived analyses,
final paper figures, and a technology atlas. Its data card documents the directory
layout, study conditions, engine revision, checksums, and worked examples.

## Download

Install the Hugging Face command-line client and download into a directory outside
this source checkout:

```bash
python -m pip install --upgrade huggingface_hub
hf download lamm-mit/swarmworld-data \
  --repo-type dataset \
  --local-dir ../swarmworld-data
```

To fetch only one authoritative trace:

```bash
hf download lamm-mit/swarmworld-data \
  studies/study_800tick_n050_full_independent/llm-full-n-50-seed-3201.jsonl.gz \
  --repo-type dataset \
  --local-dir ../swarmworld-data
```

The source release does not need the dataset for unit tests or scripted simulation.
After downloading a trace, verify it with the compatible engine revision:

```bash
biofoundry replay \
  ../swarmworld-data/studies/study_800tick_n050_full_independent/\
llm-full-n-50-seed-3201.jsonl.gz
```

Do not copy the dataset into this repository. `data_share/`, generated run outputs,
reports, and paper-figure directories are ignored and rejected by the release
verifier. This separation keeps the software clone small and gives the scientific
data its own versioned manifest.

## Code/data compatibility

The released paper studies use authoritative engine revision 9. Each trace and study
manifest records the relevant engine revision, resolved protocol, source commit, and
integrity information. Exact replay is defined only for a compatible engine revision;
older or mismatched traces receive the more limited integrity behavior documented in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Licensing and citation

The dataset data card states its license and citation metadata. Cite the software and
dataset separately so the exact versions used in an analysis can be identified.
