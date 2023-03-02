# music-ner-eacl2023

This repository provides Python code to reproduce the experiments from the article **A Human Subject Study of Named Entity Recognition (NER) in Conversational Music Recommendation Queries**, accepted for publication to [**EACL 2023**](https://2023.eacl.org).

## Installation

```sh
git clone git@github.com:deezer/music-ner-eacl2023.git
cd music-ner-eacl2023
```

## Setup

Build the docker image and run it in a container while launching an interactive bash session (the current docker image requires a CUDA-capable GPU):

```sh
$ make build
$ make run-bash
```

## Experiments

### Data statistics and preparation

Print the data statistics shown in `Table 2` of the paper:

```bash
poetry run python3 music-ner/datasets/stats.py --data_dir=data/dataset1
poetry run python3 music-ner/datasets/stats.py --data_dir=data/dataset2
poetry run python3 music-ner/datasets/stats.py --data_dir=data/dataset3
poetry run python3 music-ner/datasets/stats.py --data_dir=data/dataset4
```

Prepare ground-truth sets with `seen` and `rare / unseen` entities:
```bash
poetry run python3 music-ner/datasets/create_seen_rare_ds.py --data_dir data/dataset1/ --th_seen=1 --th_rare_unseen=0
poetry run python3 music-ner/datasets/create_seen_rare_ds.py --data_dir data/dataset2/ --th_seen=1 --th_rare_unseen=0
poetry run python3 music-ner/datasets/create_seen_rare_ds.py --data_dir data/dataset3/ --th_seen=1 --th_rare_unseen=0
poetry run python3 music-ner/datasets/create_seen_rare_ds.py --data_dir data/dataset4/ --th_seen=1 --th_rare_unseen=0
```

### Fine-tuning

*Note: some small variations between different runs, hence from the exact scores reported in the paper, could exist but with no statistically significant differences.*

Fine-tune multiple transformers (`BERT`, `RoBERTa` and `MPNet`) to perform music NER and print results (`Table 4`):
```bash
./music-ner/scripts/run_ner_model_selection.sh
poetry run python3 music-ner/tables-and-stats/transformer_baselines.py --results_dir output
```

Fine-tune `BERT` to perform music NER, export `human` annotation results in the same `json` format as the one produced by transformers, and print results (`Tables 5` and `6`):
```bash
./music-ner/scripts/run_ner.sh
poetry run python3 music-ner/src/compute_human_performance.py --data_dir data/dataset1 --output_dir output/dataset1
poetry run python3 music-ner/src/compute_human_performance.py --data_dir data/dataset2 --output_dir output/dataset2
poetry run python3 music-ner/src/compute_human_performance.py --data_dir data/dataset3 --output_dir output/dataset3
poetry run python3 music-ner/src/compute_human_performance.py --data_dir data/dataset4 --output_dir output/dataset4
poetry run python3 music-ner/tables-and-stats/human_vs_bert.py --results_dir output
```

Run experiments for `seen` and `rare / unseen` ground-truth sets and print results (`Table 7`):
```bash
./music-ner/scripts/run_ner_seen_ents.sh
./music-ner/scripts/run_ner_rare_unseen_ents.sh
poetry run python3 music-ner/tables-and-stats/seen_vs_unseen.py --results_dir output
```

Reproduce `Figure 1` with the detailed error analysis for `BERT` and `human` predictors:
```bash
poetry run python3 music-ner/tables-and-stats/graph_error_analysis.py --results_dir output
```

## Paper

Please cite our paper if you use this data or code in your work:
```
@InProceedings{Epure2023,
  title={A Human Subject Study of Named Entity Recognition (NER) in Conversational Music Recommendation Queries},
  author={Epure, Elena and Hennequin, Romain},
  booktitle={Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics (EACL)},
  month={May},
  year={2023}
}
```

