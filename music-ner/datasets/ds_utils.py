import pandas as pd


def read_sents(bio_file):
    """
    Read a BIO file
    Return sentences as a dictionary with sentence's text as key
    and (token, tag) pairs as value
    """
    sents = {}
    with open(bio_file, "r") as _:
        lines = [line.replace("\n", "") for line in _.readlines()]
        sent = []
        for line in lines:
            if line == "":
                key = " ".join([t[0] for t in sent])
                sents[key] = sent
                sent = []
            else:
                token, tag = line.split("\t")
                sent.append((token, tag))
    return sents


def save_sents(sents, bio_file):
    """
    Save sentences (lists of (token, tag) pairs) in a BIO file
    """
    with open(bio_file, "w") as _:
        for sent in sents.values():
            for token_tag in sent:
                _.write("\t".join(token_tag) + "\n")
            _.write("\n")


def entities(sents):
    """
    Extract all entities grouped by type from sentences
    given as (token, tag) pair lists
    """
    ents = {}
    for sent in sents:
        ent = ""
        etype = None
        for token, tag in sent:
            if tag.startswith("B-"):
                if ent and etype:
                    if etype not in ents:
                        ents[etype] = []
                    ents[etype].append(ent)
                    ent = ""
                    etype = None
                etype = tag[2:]
                ent = token
            elif tag.startswith("I-"):
                ent += " " + token
            else:
                continue
        if ent and etype:
            if etype not in ents:
                ents[etype] = []
            ents[etype].append(ent)
    return ents


def mask_ents(sents, keep_ents):
    """
    Change tags to O for all entities apart from those in keep_ents
    keep_ents: dict with sentence's text as key and entities as values
    sents: dict with sentence's text as key and (token, tag) pairs as value
    Return a new dictionary of sentences with the updated tags
    """
    new_sents = {}
    for sent in sents:
        new_sents[sent] = []
        # print('before', sents[sent])
        if sent in keep_ents:
            sent_words = sent.split()
            ents = keep_ents[sent]
            no_change_indices = set()
            # print(ents)
            for ent in ents:
                ent_words = ent.split()
                indices = find_indices(ent_words, sent_words)
                for start, stop in indices:
                    if start == -1:
                        print("Issue with entity ", ent)
                        continue
                    else:
                        for i in range(start, stop):
                            no_change_indices.add(i)
            for i in range(len(sents[sent])):
                if i in no_change_indices:
                    new_sents[sent].append(sents[sent][i])
                else:
                    new_sents[sent].append((sents[sent][i][0], "O"))
        else:
            for i in range(len(sents[sent])):
                new_sents[sent].append((sents[sent][i][0], "O"))
        # print('after', new_sents[sent])
        # print('\n')
    return new_sents


def find_indices(ent_words, sent_words):
    """
    Find the start and end indices of an entity in a sentence
    The entity and sentence are given as word lists,
    but the function works directly on strings too
    """
    indices = []
    start_index = -1
    stop_index = 0
    for i in range(len(sent_words)):
        if stop_index == len(ent_words):
            # found an occurence, add it to the list
            indices.append((start_index, start_index + len(ent_words)))
            # reset stop index
            stop_index = 0
        if sent_words[i] == ent_words[stop_index]:
            if stop_index == 0:
                start_index = i
            stop_index += 1
        else:
            stop_index = 0
    # check if there is one final occurence not saved
    if stop_index == len(ent_words):
        indices.append((start_index, start_index + len(ent_words)))
    return indices


def get_ents_seen_by_humans(data_dir):
    """
    Return all entities known by human annotators on this dataset
    """
    seen_ents = set()
    for i in range(1, 4):
        f = f"{data_dir}/annotator{i}.csv"
        adf = pd.read_csv(f, index_col=None)
        adf = adf[adf.label.isin(["Artist_known", "WoA_known"])]
        for r in adf.to_records():
            entity = r[2][r[3] : r[4]]
            seen_ents.add(entity)
    return seen_ents
