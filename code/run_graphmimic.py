import argparse
import logging
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm, trange

import transformers
from transformers import AdamW, AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup
from modeling_bert_codes import BertForSequenceClassification
from modeling_electra_codes import ElectraForSequenceClassification

from data_utils import MimicProcessor, InputFeatures, InputExample
from utils import set_seed, compute_metrics, ArgParser, aggregate_predictions, get_summary

from tensorboardX import SummaryWriter
import wandb

wandb.init(project='graphmimic', entity='dc925')

logger = logging.getLogger(__name__)


def train(args, train_dataset, model, processor, tokenizer):
    tb_writer = SummaryWriter()
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    args.eval_steps = t_total // args.num_train_epochs // 4
    args.warmup_steps = t_total // 20

    if args.codes_attention:
        codes_params = [{'params': [p for n, p in model.named_parameters(
        ) if 'graph' in n], 'lr': args.codes_learning_rate}]
        base_params = [{'params': [p for n, p in model.named_parameters(
        ) if 'graph' not in n], 'lr': args.learning_rate}]
        codes_optimizer = AdamW(
            codes_params, lr=args.codes_learning_rate, eps=args.adam_epsilon)
        base_optimizer = AdamW(
            base_params, lr=args.learning_rate, eps=args.adam_epsilon)
        codes_scheduler = get_linear_schedule_with_warmup(
            codes_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
        base_scheduler = get_linear_schedule_with_warmup(
            base_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    else:
        optimizer = AdamW(model.parameters(),
                          lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(epochs_trained, int(
        args.num_train_epochs), desc="Epoch")
    set_seed(args)

    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()

    wandb.watch(model, log='all', log_freq=args.eval_steps)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1],
                      "token_type_ids": batch[2], "labels": batch[3],
                      'codes_attention_mask': batch[4],
                      'output_attentions': True}

            if args.fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(**inputs)
                    loss = outputs[0]
            else:
                outputs = model(**inputs)
                loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)

                if args.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if args.codes_attention:
                        codes_optimizer.step()
                        base_optimizer.step()
                        codes_scheduler.step()
                        base_scheduler.step()
                        codes_optimizer.zero_grad()
                        base_optimizer.zero_grad()
                    else:
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                # scheduler.step()
                # optimizer.zero_grad()
                model.zero_grad()
                global_step += 1

                if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    if args.evaluate_during_training:
                        results = evaluate(args, model, processor, tokenizer)
                        for k, v in results.items():
                            tb_writer.add_scalar(
                                "eval_{}".format(k), v, global_step)
                            wandb.log({'eval_{}'.format(k): v},
                                      step=global_step, commit=False)

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    running_loss = (tr_loss - logging_loss) / \
                        args.logging_steps
                    if args.codes_attention:
                        codes_lr = codes_scheduler.get_lr()[0]
                        base_lr = base_scheduler.get_lr()[0]
                        wandb.log({'lr': base_lr, 'codes_lr': codes_lr,
                                   'loss': running_loss}, step=global_step)
                    else:
                        lr = scheduler.get_lr()[0]
                    # tb_writer.add_scalar('lr', lr, global_step)
                    # tb_writer.add_scalar('loss', running_loss, global_step)
                        wandb.log({'lr': lr, 'loss': running_loss},
                                  step=global_step)

                    logging_loss = tr_loss

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(
                        args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (model.module if hasattr(
                        model, 'module') else model)
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(
                        output_dir, "training_args.bin"))
                    logger.info(
                        "saving model checkpoint to {}".format(output_dir))

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, processor, tokenizer, mode='valid', prefix=""):
    results = {}
    eval_dataset = load_and_cache_examples(
        args, processor, tokenizer, mode=mode)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    for batch in tqdm(eval_dataloader, desc="Evaluation"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1],
                      "token_type_ids": batch[2], "labels": batch[3],
                      'codes_attention_mask': batch[4],
                      'output_attentions': True}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    # preds = np.argmax(preds, axis=1)

    # do aggregate metrics here
    data_path = os.path.join(args.data_dir, '{}.csv'.format(mode))
    agg_preds, agg_labels = aggregate_predictions(preds, data_path)
    aggregate_result = compute_metrics(agg_preds, agg_labels)
    renamed_aggregate_result = {}
    for k, v in aggregate_result.items():
        renamed_aggregate_result['{}_{}'.format('agg', k)] = v

    result = compute_metrics(preds, out_label_ids)
    result.update({'loss': eval_loss})
    result.update(renamed_aggregate_result)
    results.update(result)

    output_eval_file = os.path.join(
        args.output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, 'w') as writer:
        logger.info("***** Eval Results {} *****".format(mode))
        for key in sorted(result.keys()):
            logger.info(" {} = {}".format(key, str(result[key])))
            writer.write(" {} = {}\n".format(key, str(result[key])))
    return results


def load_and_cache_examples(args, processor, tokenizer, mode="train"):
    cached_features_file = os.path.join(args.data_dir, "cached_{}_{}_{}_{}".format(
        mode,
        list(filter(None, args.model_name_or_path.split("/"))).pop(),
        args.task,
        str(args.max_seq_length)
    ))
    if args.include_codes:
        cached_features_file = cached_features_file + '_codes'
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("loading features from cached file {}".format(
            cached_features_file))
        features = torch.load(cached_features_file)
    else:
        logger.info(
            "Creating features from dataset file at {}".format(args.data_dir))
        label_list = processor.get_labels()
        examples = processor.get_examples(args.data_dir, mode)
        features = convert_examples_to_features(
            examples, tokenizer, max_length=args.max_seq_length, label_list=label_list, include_codes=args.include_codes
        )

        logger.info("Saving features into cached file {}".format(
            cached_features_file))
        torch.save(features, cached_features_file)

    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    all_codes_attention_mask = torch.tensor(
        [f.codes_attention_mask for f in features], dtype=torch.long)
    dataset = TensorDataset(
        all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_codes_attention_mask)
    return dataset


def convert_examples_to_features(examples, tokenizer, max_length=None, label_list=None, include_codes=None):
    if max_length is None:
        max_length = tokenizer.max_len

    label_map = {label: i for i, label in enumerate(label_list)}

    labels = [label_map[example.label] for example in examples]

    if include_codes:
        for example in examples:
            # example.codes = ['ICD'+c for c in example.codes]
            example.text = " ".join(example.codes) + " [SEP] " + example.text

    batch_encoding = tokenizer([example.text for example in examples],
                               max_length=max_length, padding='max_length', truncation=True)
    codes_attention_masks = get_codes_attention_mask(
        examples, max_length=max_length)
    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        inputs['codes_attention_mask'] = codes_attention_masks[i]
        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("hadm_id: {}".format(example.hadm_id))
        logger.info('codes: {}'.format(example.codes))
        logger.info("text: {}".format(example.text))
        logger.info('tokens: {}'.format(
            tokenizer.convert_ids_to_tokens(features[i].input_ids)))
        logger.info("features: {}".format(features[i]))
    return features


def get_codes_attention_mask(examples, max_length=None):
    num_codes = [len(example.codes) for example in examples]
    code_attention_masks = [
        [1]*(1+num_code)+[0]*(max_length-1-num_code) for num_code in num_codes]
    return code_attention_masks


def main():
    args = ArgParser().parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    logger.warning(
        "device: %s, n_gpu: %s, 16-bits training: %s",
        args.device,
        args.n_gpu,
        args.fp16,
    )

    set_seed(args)

    processor = MimicProcessor()

    label_list = processor.get_labels()
    num_labels = len(label_list)

    config = AutoConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                        num_labels=num_labels,
                                        finetuning_task=args.task,
                                        cache_dir=args.cache_dir)
    args.model_type = config.model_type
    config.codes_attention = args.codes_attention
    config.threshold = args.threshold
    logger.info("CODES ATTENTION IS {}".format(args.codes_attention))
    logger.info("INCLUDE CODES IS {}".format(args.include_codes))
    logger.info("PRETRAINED ICD IS {}".format(args.pretrained_icd))
    logger.info("ONLY CODES IS {}".format(args.only_codes))
    config.only_codes = args.only_codes
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir
    )
    model_options = {'bert': BertForSequenceClassification,
                     'electra': ElectraForSequenceClassification}
    model_class = model_options[args.model_type]

    model = model_class.from_pretrained(
        args.model_name_or_path, config=config, cache_dir=args.cache_dir)

    # Add ICD codes as new tokens to tokenizer
    if args.include_codes:
        icd_codes_mortality = pd.read_csv(
            '/home/dc925/project/data/graphmimic/mortality/icd_codes_mortality.txt', header=None)
        icd_codes_readmission = pd.read_csv(
            '/home/dc925/project/data/graphmimic/readmission/icd_codes_readmission.txt', header=None)
        icd_codes_mortality = icd_codes_mortality[0].tolist()
        icd_codes_readmission = icd_codes_readmission[0].tolist()
        icd_codes = set(icd_codes_mortality + icd_codes_readmission)
        icd_codes = sorted(icd_codes)
        icd_codes_tokens = ['ICD'+c for c in icd_codes]
        num_added_tokens = tokenizer.add_tokens(icd_codes_tokens)
        logger.info('we have added {} tokens'.format(num_added_tokens))
        model.resize_token_embeddings(len(tokenizer))

    if args.pretrained_icd:
        assert args.include_codes
        # read in kge and entities.tsv
        kge = np.load(
            '/home/dc925/project/graphmimic/ckpts/RotatE_ICD9_2/ICD9_RotatE_entity.npy')
        entities = pd.read_csv(
            '/home/dc925/project/data/graphmimic/UMLS/ICD_KG/entities.tsv', sep='\t', header=None)
        entities.columns = ['ID', 'ICD']
        icd2id = pd.Series(entities['ID'].values,
                           index=entities['ICD']).to_dict()
        id2icd = {v: k for k, v in icd2id.items()}

        broad_idx = [icd2id[c] for c in icd_codes]
        broad_kge = kge[broad_idx]
        assert broad_kge.shape[1] == config.embedding_size

        with torch.no_grad():
            embeddings = model.get_input_embeddings()
            embeddings.weight[-num_added_tokens:, :] = torch.tensor(broad_kge)

    model.to(args.device)
    logger.info("Training/evaluation parameters {}".format(args))
    get_summary(model)

    if args.do_train:

        train_dataset = load_and_cache_examples(args, processor, tokenizer)
        global_step, tr_loss = train(
            args, train_dataset, model, processor, tokenizer)
        logger.info("global_step = {}, average loss = {}".format(
            global_step, tr_loss))

        logger.info("Saving model checkpoint to {}".format(args.output_dir))
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        model = model_class.from_pretrained(
            args.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
        model.to(args.device)

    results = {}
    if args.do_eval:
        result = evaluate(args, model, processor, tokenizer)
        result = dict((k + "'_{}".format(global_step), v)
                      for k, v in result.items())
        results.update(result)
    if args.do_test:
        result = evaluate(args, model, processor, tokenizer, mode='test')
        result = dict((k + "'_{}".format(global_step), v)
                      for k, v in result.items())
        results.update(result)
    return results


if __name__ == "__main__":
    main()
