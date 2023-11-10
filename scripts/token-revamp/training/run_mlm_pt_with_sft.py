import os
from argparse import ArgumentParser, Namespace
from glob import glob
from pathlib import Path
from typing import Tuple, Optional

import datasets
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoConfig, 
    AutoTokenizer,
    AutoModelWithLMHead,
    XLMRobertaTokenizer,
    XLMRobertaForMaskedLM
)
from tqdm import tqdm


def run_parser() -> Tuple[Namespace, Namespace, Namespace]:
    parser = ArgumentParser()
    train_parser = parser.add_argument_group("training")
    model_parser = parser.add_argument_group("model")
    data_parser = parser.add_argument_group("data")
    
    # add arguments
    # train
    train_parser.add_argument("--output_dir", type=str, default="./output")
    train_parser.add_argument("--overwrite_output_dir", action="store_true", help="overwrite output directory")
    train_parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="resume from checkpoint")
    train_parser.add_argument("--seed", type=int, default=42, help="random seed")
    train_parser.add_argument("--debug_mode", action="store_true", help="debug mode")
    train_parser.add_argument("--do_train", action="store_true", help="do train")
    train_parser.add_argument("--do_eval", action="store_true", help="do eval")

    # model
    model_parser.add_argument("--model_name_or_path", type=str, required=True, help="model name or path")
    model_parser.add_argument("--tokenizer_name_or_path", type=str, default=None, help="tokenizer name or path")
    model_parser.add_argument("--model_revision", type=str, default="main", help="model revision")
    model_parser.add_argument("--config_name", type=str, default=None, help="config name")
    model_parser.add_argument("--tokenizer_name", type=str, default=None, help="tokenizer name")
    model_parser.add_argument("--cache_dir", type=str, default="./cache", help="cache dir")
    model_parser.add_argument("--use_fast_tokenizer", action="store_true", help="use fast tokenizer")
    model_parser.add_argument("--torch_dtype", type=str, default="float32", help="torch dtype")

    # data
    data_parser.add_argument("--dataset_dir", type=str, required=True, help="Path to dataset containing text files")
    data_parser.add_argument("--data_cache_dir", type=str, default="./data_cache", help="data cache dir, store tokenized data")
    data_parser.add_argument("--max_seq_length", type=int, default=512, help="max sequence length")
    data_parser.add_argument("--max_train_samples", type=int, default=None, help="max train samples")
    data_parser.add_argument("--max_eval_samples", type=int, default=None, help="max eval samples")
    data_parser.add_argument("--validation_split_percentage", type=float, default=0.1, help="validation split percentage")
    data_parser.add_argument("--preprocessing_num_workers", type=int, default=4, help="preprocessing num workers")

    # parse arguments
    args = parser.parse_args()
    training_args = Namespace()
    model_args = Namespace()
    data_args = Namespace()

    for k, v in args.__dict__.items():
        if k in list(map(lambda x: x.dest, train_parser._group_actions)):
            setattr(training_args, k, v)
        elif k in list(map(lambda x: x.dest, model_parser._group_actions)):
            setattr(model_args, k, v)
        elif k in list(map(lambda x: x.dest, data_parser._group_actions)):
            setattr(data_args, k, v)
        else:
            raise ValueError(f"Argument {k} is not defined")

    return training_args, model_args, data_args


def get_last_checkpoint(output_dir: str) -> Optional[str]:
    output_dir = Path(output_dir)
    checkpoints = list(output_dir.glob("checkpoint-*"))
    if len(checkpoints) == 0:
        return None
    get_ckpt_step = lambda ckpt: int(str(ckpt).split("-")[-1])
    last_checkpoint = max(checkpoints, key=get_ckpt_step)
    return str(last_checkpoint)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # parser
    training_args, model_args, data_args = run_parser()

    # detecting last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and \
        training_args.do_train \
        and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        # if output_dir already exists, we don't want to overwrite it
        if last_checkpoint is not None and \
            training_args.resume_from_checkpoint is None:
            # last_checkpoint is detected and resume_from_checkpoint is not given
            # raise error to avoid overwrite
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            )
        elif (last_checkpoint is not None) and \
            training_args.resume_from_checkpoint is None:
            # last_checkpoint is detected and resume_from_checkpoint is given
            print(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # set seed
    set_seed(training_args.seed)

    # load model
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        # no config is give, raise error
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --config_name."
        )
    
    # load tokenizer
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = XLMRobertaTokenizer.from_pretrained(model_args.tokenizer_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name."
        )
    
    # preprocess datasets
    mlm_datasets = []
    files = [
        os.path.relpath(file, data_args.dataset_dir)
        for file in glob(f"{data_args.dataset_dir}/**/*.txt", recursive=True)]
    print(f"Found {len(files)} files in {data_args.dataset_dir}")
    # debug mode only use one file
    if training_args.debug_mode:
        files = files[:1]
    for idx, file in tqdm(enumerate(files), desc="Preprocessing datasets"):
        data_file = os.path.join(data_args.dataset_dir, file)
        filename = "".join(file.split(".")[:-1]) # remove extension
        cache_path = os.path.join(data_args.data_cache_dir, filename)
        os.makedirs(cache_path, exist_ok=True)

        try:
            processed_dataset = datasets.load_from_disk(cache_path, keep_in_memory=False)
            print(f"{file} has been loaded from disk")
        except:
            cache_dir = os.path.join(data_args.data_cache_dir, filename+"_text")
            raw_dataset = load_dataset("text", data_files=data_file, cache_dir=cache_dir, keep_in_memory=False)
            print(f"{file} has been loaded")

            processed_dataset = raw_dataset.map(
                lambda example: tokenizer.encode_plus(
                    example["text"], 
                    truncation=True, 
                    padding="max_length",
                    max_length=data_args.max_seq_length,  # wangchanberta max length is 412
                ),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=["text"],
                load_from_cache_file=False,
                keep_in_memory=False,
                cache_file_names={k: os.path.join(cache_dir, "tokenized.arrow") for k in raw_dataset.keys()},
                desc="Running tokenizer on dataset",
            )  
            processed_dataset.save_to_disk(cache_path)
        
        if idx == 0:
            # first dataset
            mlm_datasets = processed_dataset["train"]
        else:
            # other dataset, concatenate
            assert mlm_datasets.features.type == processed_dataset["train"].features.type, \
                "All datasets must have the same type of features"
            mlm_datasets = datasets.concatenate_datasets([mlm_datasets, processed_dataset["train"]])

    mlm_datasets = mlm_datasets.train_test_split(
        test_size=data_args.validation_split_percentage,
        shuffle=True,
        seed=training_args.seed,
    )

    if training_args.do_train:
        train_dataset = mlm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        print(f"Num train_samples  {len(train_dataset)}")
        print("training example:")
        print(tokenizer.decode(train_dataset[0]["input_ids"]))
    if training_args.do_eval:
        eval_dataset = mlm_datasets["test"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        print(f"Num eval_samples  {len(eval_dataset)}")
        print("eval example:")
        print(tokenizer.decode(eval_dataset[0]["input_ids"]))

    # load model
    if model_args.model_name_or_path:
        model = XLMRobertaForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir
        )
    else:
        model = AutoModelWithLMHead.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        print(f"Training new model from scratch. Number of parameters: {n_params/2**20:.2f}M parameters")

    model_vocab_size = model.get_input_embeddings().weight.shape[0]
    print(f"Model vocab size before applying token revamp: {model_vocab_size}")

    # apply token revamp
    model.resize_token_embeddings(len(tokenizer))
    model_vocab_size = model.get_input_embeddings().weight.shape[0]
    print(f"Model vocab size after applying token revamp: {model_vocab_size}")


if __name__ == "__main__":
    main()
