# MusicRecoNER

`MusicRecoNER` is a corpus of noisy complex natural language queries for music recommendation collected from human-human conversations in English, but which simulates human-music assistant interactions, annotated with `Artist` and `WoA` (work of art) entities. It consists of four different datasets corresponding to the various annotation groups (`dataset1` - `DS1`, `dataset2` - `DS2`, `dataset3` - `DS3`, and `dataset4` - `Trial`) as described in the paper. Each dataset folder contains the following files / folders:

- the original and preprocessed queries (`queries.csv`)

- the annotations produced by each annotator in the csv and bio formats (e.g. `annotator1.csv` and `annotator1.bio`)

- the ground-truth in bio format (`ground-truth.bio`)

- the ground-truth entities linked to wikidata and their computed exposure (`ground-truth_linked.csv`)

- the test file (`test.bio`) which is the same as `ground-truth.bio`

- the train file (`train.bio`) which is composed by concatenating all the other ground-truth bio files corresponding to the other annotation groups (e.g. for `dataset1`, we concatenate the `ground-truth.bio` files from `dataset2`, `dataset3`, and `dataset4`)

- the folders `seen` and `rare_unseen` contain the test and train files modified such that only the entities considered seen or rare or unseen respectively are kept and all the other entities are annotated with the `O` tag. As a reminder, we compute the recall only for the selected/unmasked entities.