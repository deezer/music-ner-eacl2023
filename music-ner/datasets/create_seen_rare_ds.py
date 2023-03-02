import argparse
import os
import shutil
import sys

import ds_utils as dsu
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", dest="data_dir", type=str, help="Data directory", required=True
    )
    parser.add_argument(
        "--th_seen",
        dest="th_seen",
        type=int,
        default=1,
        help="Threshold for seen entities",
        required=True,
    )
    parser.add_argument(
        "--th_rare_unseen",
        dest="th_rare_unseen",
        type=int,
        default=0,
        help="Threshold for rare or unseen entities",
        required=True,
    )
    args = parser.parse_args()
    data_dir = args.data_dir

    data_fpath = os.path.join(data_dir, "train.bio")
    if not os.path.isfile(data_fpath):
        print("{} file not found".format(data_fpath))
        sys.exit(1)

    # create output folder for seen entities
    out_dir_seen = data_dir + "/seen"
    if not os.path.exists(out_dir_seen):
        os.makedirs(out_dir_seen)
    # copy the training data
    shutil.copyfile(data_fpath, os.path.join(out_dir_seen, "train.bio"))

    # create output folder for unseen entities
    out_dir_rare_unseen = data_dir + "/rare_unseen"
    if not os.path.exists(out_dir_rare_unseen):
        os.makedirs(out_dir_rare_unseen)
    # copy the training data
    shutil.copyfile(data_fpath, os.path.join(out_dir_rare_unseen, "train.bio"))

    # read the csv containing the linked entities and their exposure
    ents_fpath = os.path.join(data_dir, "ground-truth_linked.csv")
    if not os.path.isfile(ents_fpath):
        print("{} file not found".format(ents_fpath))
        sys.exit(1)
    df = pd.read_csv(ents_fpath)
    df.fillna("", inplace=True)

    # read the original test file that will be changed
    test_fpath = os.path.join(data_dir, "test.bio")
    if not os.path.isfile(test_fpath):
        print("{} file not found".format(test_fpath))
        sys.exit(1)
    sents = dsu.read_sents(test_fpath)

    # artificially change the exposure to max for all entities
    # that are found in the train set
    train_sents = dsu.read_sents(data_fpath)
    train_ents = dsu.entities(train_sents.values())
    test_ents = dsu.entities(sents.values())
    test_ents_all = set(test_ents["Artist"] + test_ents["WoA"])
    common_ents = set()
    for etype in train_ents:
        common_ents.update(set(train_ents[etype]).intersection(set(test_ents[etype])))
    df.loc[df.mention.isin(common_ents), "exposure"] = df["exposure"].max() + 1

    # find all entities seen by humans in order to remove those from unseen
    seen_humans = dsu.get_ents_seen_by_humans(data_dir)

    # consider the entities seen during pre-training (exposure > threshold)
    seen = {}
    tuples = df[df.exposure > args.th_seen][["query", "mention"]].itertuples(
        index=False
    )
    for query, ent in tuples:
        if query not in seen:
            seen[query] = []
        seen[query].append(ent)
    print(
        "Seen - Number of unique queries with entities before masking ",
        len(set(df["query"].to_list())),
    )
    # mask entities not among the queries in seen (their tag becomes "O")
    seen_sents = dsu.mask_ents(sents, seen)
    print("Number of queries with seen entities", len(seen))
    dsu.save_sents(seen_sents, os.path.join(out_dir_seen, "test.bio"))

    # consider the entities seen during pre-training (exposure > threshold)
    rare_unseen = {}
    common_ents.update(seen_humans)
    # remove from the dataset all entities seen either by model during fine-tuning or acknowledged seen by humans
    df = df[~df.mention.isin(common_ents)]
    print(
        "Rare / Unseen - Number of unique queries with entities before masking ",
        len(set(df["query"].to_list())),
    )
    # unseen or rare entities are those which could not be linked or which were linked but rare
    df = df[(df.wiki_name == "") | (df.exposure <= args.th_rare_unseen)]
    tuples = df[["query", "mention"]].itertuples(index=False)
    for query, ent in tuples:
        if query not in rare_unseen:
            rare_unseen[query] = []
        rare_unseen[query].append(ent)
    print("Number of unique queries with rare unseen entities ", len(rare_unseen))
    # print(df['query'].unique())
    # mask entities not among the queries in rare_unseen (their tag becomes "O")
    rare_unseen_sents = dsu.mask_ents(sents, rare_unseen)
    dsu.save_sents(rare_unseen_sents, os.path.join(out_dir_rare_unseen, "test.bio"))
