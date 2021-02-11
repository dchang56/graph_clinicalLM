import csv
from typing import Optional, List, Union
from dataclasses import dataclass
import logging
import os
import ast


@dataclass
class InputExample:
    hadm_id: int
    text: str
    codes: Optional[List[str]] = None
    label: Optional[str] = None


@dataclass(frozen=True)
class InputFeatures:
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    codes_attention_mask: Optional[List[int]] = None


class MimicProcessor:
    def get_examples(self, data_dir, mode):
        examples = []
        input_file = os.path.join(data_dir, "{}.csv".format(mode))
        with open(input_file, "r", encoding="utf-8-sig") as f:
            lines = list(csv.reader(f, delimiter="\t"))
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            hadm_id = line[0]
            text = line[1]
            codes = ast.literal_eval(line[2])
            label = str(int(float(line[3])))
            examples.append(InputExample(hadm_id=hadm_id,
                                         text=text, codes=codes, label=label))

        return examples

    def get_labels(self):
        return ["0", "1"]
