import random
import numpy as np
import torch
import os
import argparse
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    auc,
    roc_curve,
    precision_recall_curve)
import pandas as pd


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(preds, labels):
    metrics = {}
    preds = np.argmax(preds, axis=1)
    accuracy = (preds == labels).mean()
    metrics['accuracy'] = accuracy
    # auprc
    # precisions, recalls, thresholds = precision_recall_curve(labels, preds)
    # auc_pr = auc(recalls, precisions)
    # metrics['AUCPR'] = auc_pr
    # auroc
    # fpr, tpr, thresholds = roc_curve(labels, preds)
    # auc_roc = auc(fpr, tpr)
    # metrics['AUROC'] = auc_roc
    # f1 score, precision, recall
    # precision, recall, fscore, support = precision_recall_fscore_support(labels, preds, average='weighted')
    # metrics['precision'] = precision
    # metrics['recall'] = recall
    # metrics['fscore'] = fscore
    return metrics


def aggregate_predictions(preds, data_path):
    df = pd.read_csv(data_path, sep='\t')
    df['pred_0'] = preds[:, 0]
    df['pred_1'] = preds[:, 1]
    df_sort = df.sort_values(by=['HADM_ID'])
    temp_0 = (df_sort.groupby(['HADM_ID'])['pred_0'].agg(max) + df_sort.groupby(['HADM_ID'])[
              'pred_0'].agg(sum)/2) / (1 + df_sort.groupby(['HADM_ID'])['pred_0'].agg(len)/2)
    temp_1 = (df_sort.groupby(['HADM_ID'])['pred_1'].agg(max) + df_sort.groupby(['HADM_ID'])[
              'pred_1'].agg(sum)/2) / (1 + df_sort.groupby(['HADM_ID'])['pred_1'].agg(len)/2)
    l = df_sort.groupby(['HADM_ID'])['OUTPUT_LABEL'].agg(np.min).values
    t0 = temp_0.to_numpy()
    t1 = temp_1.to_numpy()
    agg_preds = np.stack([t0, t1], axis=1)
    return agg_preds, l


def get_summary(model):
    total_params = 0
    for name, param in model.named_parameters():
        shape = param.shape
        param_size = 1
        for dim in shape:
            param_size *= dim
        print(name, shape, param_size)
        total_params += param_size
    print(total_params)


class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__()

        self.add_argument("--data_dir", default=None, type=str, required=False)
        self.add_argument("--model_name_or_path",
                          default=None, type=str, required=False)
        self.add_argument("--output_dir", default=None,
                          type=str, required=False)

        # Other parameters
        self.add_argument("--config_name", default="", type=str,
                          help="Pretrained config name or path if not the same as model_name")
        self.add_argument("--tokenizer_name", default="", type=str,
                          help="Pretrained tokenizer name or path if not the same as model_name")
        self.add_argument("--cache_dir", default=None, type=str)
        self.add_argument("--task_name", default="mortality", type=str)

        self.add_argument("--max_seq_length", default=128, type=int,
                          help="The maximum total input sequence length after tokenization. Sequences longer ""than this will be truncated, sequences shorter will be padded.")
        self.add_argument("--do_train", action="store_true",
                          help="Whether to run training.")
        self.add_argument("--do_eval", action="store_true",
                          help="Whether to run eval on the test set.")
        self.add_argument("--do_test", action='store_true')
        self.add_argument("--evaluate_during_training", action="store_true",
                          help="Rul evaluation during training at each logging step.")
        self.add_argument("--do_lower_case", action="store_true",
                          help="Set this flag if you are using an uncased model.")

        self.add_argument("--include_codes", action='store_true')
        self.add_argument('--codes_attention', action='store_true')
        self.add_argument('--threshold', default=0.3, type=float)
        self.add_argument('--only_codes', action='store_true')
        self.add_argument('--pretrained_icd', action='store_true')
        self.add_argument("--task", type=str, required=True)

        self.add_argument("--per_gpu_train_batch_size", default=16,
                          type=int, help="Batch size per GPU/CPU for training.")
        self.add_argument("--per_gpu_eval_batch_size", default=16,
                          type=int, help="Batch size per GPU/CPU for evaluation.")
        self.add_argument("--gradient_accumulation_steps", type=int, default=1,
                          help="Number of updates steps to accumulate before performing a backward/update pass.")
        self.add_argument("--learning_rate", default=5e-5,
                          type=float, help="The initial learning rate for Adam.")
        self.add_argument('--codes_learning_rate', default=5e-5, type=float)
        self.add_argument("--weight_decay", default=0.0,
                          type=float, help="Weight decay if we apply some.")
        self.add_argument("--adam_epsilon", default=1e-8,
                          type=float, help="Epsilon for Adam optimizer.")
        self.add_argument("--max_grad_norm", default=1.0,
                          type=float, help="Max gradient norm.")
        self.add_argument("--num_train_epochs", default=3.0, type=float,
                          help="Total number of training epochs to perform.")
        self.add_argument("--max_steps", default=-1, type=int,
                          help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
        self.add_argument("--warmup_steps", default=0, type=int,
                          help="Linear warmup over warmup_steps.")
        self.add_argument("--logging_steps", type=int,
                          default=10, help="Log every X updates steps.")
        self.add_argument("--save_steps", type=int, default=0,
                          help="Save checkpoint every X updates steps.")
        self.add_argument("--eval_steps", type=int, default=500,
                          help="eval model every x steps")

        self.add_argument("--eval_all_checkpoints", action="store_true", help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
                          )
        self.add_argument("--no_cuda", action="store_true",
                          help="Avoid using CUDA when available")
        self.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
                          )
        self.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
                          )
        self.add_argument("--seed", type=int, default=42,
                          help="random seed for initialization")

        self.add_argument("--fp16", action="store_true", help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
                          )
        self.add_argument("--fp16_opt_level", type=str, default="O1", help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].""See details at https://nvidia.github.io/apex/amp.html",
                          )

    def parse_args(self):
        args = super().parse_args()
        return args
