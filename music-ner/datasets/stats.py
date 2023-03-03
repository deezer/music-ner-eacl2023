import argparse
import os
import sys

import pandas as pd
from ds_utils import entities, get_ents_seen_by_humans, read_sents
from tabulate import tabulate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", dest="data_dir", type=str, help="Data directory", required=True
    )
    args = parser.parse_args()

    # read train queries
    train_fpath = os.path.join(args.data_dir, "train.bio")
    if not os.path.isfile(train_fpath):
        print("{} file not found".format(train_fpath))
        sys.exit(1)
    train_sents = read_sents(train_fpath).values()

    # read test queries
    test_fpath = os.path.join(args.data_dir, "test.bio")
    if not os.path.isfile(test_fpath):
        print("{} file not found".format(test_fpath))
        sys.exit(1)
    sents = read_sents(test_fpath)
    test_sents = sents.values()

    print("Nb of queries in train", len(train_sents))
    print("Nb of queries in test", len(test_sents))

    train_ents = entities(train_sents)
    test_ents = entities(test_sents)

    all_stats = []
    for etype in train_ents:
        stats = [etype]
        stats.append(len(train_ents[etype]))
        stats.append(len(test_ents[etype]))
        unique_train = set(train_ents[etype])
        unique_test = set(test_ents[etype])
        stats.append(len(unique_train))
        stats.append(len(unique_test))
        stats.append(len(unique_train.intersection(unique_test)))
        stats.append(
            100
            * float(len(unique_train.intersection(unique_test)))
            / float(len(unique_test))
        )
        all_stats.append(stats)

    headers = [
        "Type",
        "Nb NEs train",
        "Nb NEs test",
        "Nb unique NEs train",
        "Nb unique NEs test",
        "Nb overlap",
        "% overlap test",
    ]
    print(tabulate(all_stats, headers=headers))

    # read the csv containing the linked entities and their exposure
    ents_fpath = os.path.join(args.data_dir, "ground-truth_linked.csv")
    if not os.path.isfile(ents_fpath):
        print("{} file not found".format(ents_fpath))
        sys.exit(1)
    df = pd.read_csv(ents_fpath)
    df.fillna("", inplace=True)
    test_ents_all = set(test_ents["Artist"] + test_ents["WoA"])
    print(
        "Ratio unique Artist entitites",
        round(
            len(test_ents["Artist"])
            / (len(test_ents["Artist"]) + len(test_ents["WoA"])),
            2,
        ),
    )
    print(
        "Ratio unique WoA entitites",
        round(
            len(test_ents["WoA"]) / (len(test_ents["Artist"]) + len(test_ents["WoA"])),
            2,
        ),
    )
    print(
        "Ratio queries with no entities",
        round(len(set(sents.keys()) - set(df["query"].to_list())) / len(sents), 2),
    )
    ents_per_query = df.groupby(["query"]).size().reset_index(name="counts")
    print(
        "entities per query (mean +- std)",
        round(ents_per_query["counts"].mean(), 1),
        " +- ",
        round(ents_per_query["counts"].std(), 1),
        ", min",
        ents_per_query["counts"].min(),
        ", max",
        ents_per_query["counts"].max(),
    )

    # how many are common entities between test and train
    common_ents = set()
    for etype in train_ents:
        common_ents.update(set(train_ents[etype]).intersection(set(test_ents[etype])))
    print(
        "Ratio common entities between train and test",
        round(len(common_ents) / len(test_ents_all), 2),
    )

    pretrained_seen = set(df[df.exposure > 1]["mention"].to_list()).intersection(
        test_ents_all
    )
    print(
        "Ratio entities with exposure greater than 1 in test",
        round(len(pretrained_seen) / len(test_ents_all), 2),
    )

    seen_humans = get_ents_seen_by_humans(args.data_dir)
    # how many entities declared known by people on average
    print(
        "Ratio total number of entities seen by humans",
        round(len(seen_humans.intersection(test_ents_all)) / len(test_ents_all), 2),
    )

    print("Examples of entities with high exposure")
    print(
        df[["mention", "type", "exposure"]]
        .drop_duplicates(subset=["mention"])
        .sort_values(by="exposure", ascending=False)[:30]
    )
