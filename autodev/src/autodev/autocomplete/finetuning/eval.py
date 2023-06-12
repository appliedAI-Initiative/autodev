import logging
import multiprocessing
from typing import Sequence

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from completionft.dataset import load_train_val_datasets
from completionft.model import ModelFactory

log = logging.getLogger(__name__)


class ModelPerplexityEvaluation:
    def __init__(self, lang_id: str,
            model_factory: ModelFactory,
            model_paths: Sequence[str],
            device="cuda:0",
            max_num_code_snippets=100):
        """
        :param lang_id: a language identifier (subfolder of data/ in bigcode/the-stack-dedup)
        :param model_factory: the factory with which to create models
        :param model_paths: paths with which to call model_factory in order to obtain the concrete models
        :param device: the device onto which to load the models/data
        :param max_num_code_snippets: the maximum number of code snippets/files to use for evaluation
        """
        self.lang_id = lang_id
        self.model_factory = model_factory
        self.model_paths = model_paths
        self.device = device
        self.max_num_code_snippets = max_num_code_snippets

    def run(self) -> pd.DataFrame:
        log.info("Loading dataset")
        _, dataset = load_train_val_datasets("bigcode/the-stack-dedup", f"data/{self.lang_id}",
            num_workers=multiprocessing.cpu_count())

        base_model_id = "bigcode/santacoder"
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

        log.info("Generating encodings")
        code_snippets = dataset["content"][:self.max_num_code_snippets]
        encodings = tokenizer("\n\n".join(code_snippets), return_tensors="pt")

        rows = []
        for model_path in self.model_paths:
            log.info(f"Loading model {model_path}")
            model = self.model_factory.create_model(model_path)
            model.to(self.device)

            max_length = model.config.n_positions
            stride = 512
            seq_len = encodings.input_ids.size(1)

            nlls = []
            prev_end_loc = 0
            for begin_loc in tqdm(range(0, seq_len, stride)):
                end_loc = min(begin_loc + max_length, seq_len)
                trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
                input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100

                with torch.no_grad():
                    outputs = model(input_ids, labels=target_ids)

                    # loss is calculated using CrossEntropyLoss which averages over valid labels
                    # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                    # to the left by 1.
                    neg_log_likelihood = outputs.loss

                nlls.append(neg_log_likelihood)

                prev_end_loc = end_loc
                if end_loc == seq_len:
                    break

            ppl = torch.exp(torch.stack(nlls).mean())
            log.info(f"PPL={ppl}")

            model_name = model_path.replace("checkpoints/", "")

            rows.append(dict(model=model_name, ppl=ppl.detach().item()))

            del model

        df = pd.DataFrame(rows)
        df.sort_values("ppl", ascending=True, inplace=True)
        log.info(f"Results:\n{df.to_string()}")
        return df
