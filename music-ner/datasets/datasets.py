# coding=utf-8
# Copyright 2020 HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import os
import re

import datasets

logger = datasets.logging.get_logger(__name__)

_DESCRIPTION = """MusicRecoNER dataset loader containing noisy / unformatted queries for music recommendation with annotated Artist and WoA entities"""
_DATA_FILE = "train.bio"
_TEST_FILE = "test.bio"


class MusicNERConfig(datasets.BuilderConfig):
    """BuilderConfig for Conll2003"""

    def __init__(self, **kwargs):
        """BuilderConfig forConll2003.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MusicNERConfig, self).__init__(**kwargs)


class MusicNER(datasets.GeneratorBasedBuilder):
    """MusicNER dataset."""

    BUILDER_CONFIGS = [
        MusicNERConfig(name="music-reco-ner", version=datasets.Version("1.0.0"))
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "B-Artist",
                                "I-Artist",
                                "B-WoA",
                                "I-WoA",
                            ]
                        )
                    ),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_dir = dl_manager._data_dir
        data_files = {
            "data": os.path.join(data_dir, _DATA_FILE),
            "test": os.path.join(data_dir, _TEST_FILE),
        }
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": data_files["data"]}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": data_files["test"]}
            ),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            ner_tags = []
            for line in f:
                if line == "" or line == "\n":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "ner_tags": ner_tags,
                        }
                        guid += 1
                        tokens = []
                        ner_tags = []
                else:
                    # tokens are \t separated
                    splits = re.split(r"\s+", line)
                    tokens.append(splits[0].strip())
                    ner_tags.append(splits[1].rstrip())
            # last example
            yield guid, {
                "id": str(guid),
                "tokens": tokens,
                "ner_tags": ner_tags,
            }
