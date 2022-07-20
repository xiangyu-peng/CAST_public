#Differnece of run_lm_finetuning_classifier_frozen_1by1 is file path, this is for plotto
#This file is for finetuning the gpt2 with batching
#No plotto data usage anymore
#We use the senti data obtained from website: https://archive.ics.uci.edu/ml/machine-learning-databases/00331/
#Backward the loss with reward from sentiWorldNet
#only one gpt2, no model for generating sentences for finetuning
#Edit by ..
#latest version 11/19/2019


from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os

import pickle
import random
import csv

import sys
sys.path.append(os.getcwd())

import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

# from pytorch_transformers.tokenization_gpt2 import *
import sys
import os
sys.path.append('../../CommonSense_RL/transformer')
sys.path.append('../../CommonSense_RL/transformer/pytorch_transformers')
from pytorch_transformers import (WEIGHTS_NAME, AdamW, WarmupLinearSchedule,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)

import math
logger = logging.getLogger(__name__)

#First time use the code, need to umcomment these codes below
# import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')


MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)
}


class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path='train', block_size=512):
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory, f'cached_lm_{block_size}_{filename}')
        ############ Dict file
        self.cached_dictionary = os.path.join(directory, f'cached_dict.p')

        # if os.path.exists(cached_features_file):
        #     logger.info("Loading features from cached file %s", cached_features_file)
        #     with open(cached_features_file, 'rb') as handle:
        #         self.examples = pickle.load(handle)
        #     with open(self.cached_dictionary, 'rb') as f:
        #         self.sent_dict = pickle.load(f)
        if True:
            logger.info("Creating features from dataset file at %s", directory)

            self.examples = []
            ################################
            self.raw_examples = []
            self.sent_dict = dict()
            ################################
            with open(file_path, encoding="utf-8") as f:
                for row in f.readlines():
                    # The original sentence
                    orig_sent = row[2:]
                    orig_sent = orig_sent.lstrip()
                    orig_sent = orig_sent.rstrip()
                    orig_sent = orig_sent.replace("\n", "")
                    orig_sent_copy = orig_sent.split('.')

                    orig_sent_sen1 = orig_sent_copy[0] + '.'
                    orig_sent_sen2 = orig_sent_copy[1] + '.'

                    print('orig_sent',  orig_sent)
                    # Tokenize and process the sentence
                    token_sent = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(orig_sent))
                    token_sent = tokenizer.add_special_tokens_single_sentence(token_sent)
                    # Tokenize two sentences.
                    token_sent_sen1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(orig_sent_sen1))
                    token_sent_sen1 = tokenizer.add_special_tokens_single_sentence(token_sent_sen1)
                    token_sent_sen2 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(orig_sent_sen2))
                    token_sent_sen2 = tokenizer.add_special_tokens_single_sentence(token_sent_sen2)
                    # Append the sentence to the examples
                    print('token', token_sent)
                    token_sent_sen1 = [-1 for i in range(len(token_sent_sen1))]
                    print(token_sent_sen1)
                    print(token_sent_sen2)
                    token_sent = [token_sent, token_sent_sen1+ token_sent_sen2]
                    print('token', token_sent)
                    self.examples.append(token_sent)
                    label = row[0]
                    self.sent_dict.update({tuple(token_sent[0]): int(label)})
                    # self.label_dict.update({tuple(token_sent): tuple(label)})


            logger.info("Saving features into cached file %s", cached_features_file)
            # Save the files
            with open(cached_features_file, 'wb') as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(self.cached_dictionary, 'wb') as f:
                pickle.dump(self.sent_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])


def load_and_cache_examples(args, tokenizer, evaluate=False):
    dataset = TextDataset(tokenizer, file_path=args.eval_data_file if evaluate else args.train_data_file, block_size=args.block_size)
    return dataset


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def mask_tokens(inputs, tokenizer, args):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    masked_indices = torch.bernoulli(torch.full(labels.shape, args.mlm_probability)).bool()
    labels[~masked_indices] = -1  # We only compute loss on masked tokens
    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

def train(args, train_dataset, model, tokenizer, pho):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.train_batch_size = 1
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    # Train Dictionary ################
    train_dict = train_dataset.sent_dict

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = range(int(args.num_train_epochs))
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    # print('train_dataloader', train_dataloader)

    ########### SYLVIA -- CONTEXT DICTIONARY
    context_dict = dict()
    # the context dictionary stores the context the words appear in
    ########### ==================
    import nltk
    from nltk.corpus import sentiwordnet as swn
    from nltk.stem import WordNetLemmatizer
    from nltk.stem import PorterStemmer
    from nltk.corpus.reader.wordnet import WordNetError
    for _ in train_iterator:
            # epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
            epoch_iterator = train_dataloader
            for step, batch in enumerate(epoch_iterator): #batch is tensor with inputs tokens
                inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
                inputs, labels = inputs[0][0].reshape(1,-1).to(args.device), inputs[0][1].reshape(1,-1).to(args.device)
                model.train()
                print('inputs', inputs, inputs.shape)
                print('labels,', labels, labels.shape)
                outputs = model(inputs, masked_lm_labels=labels) if args.mlm else model(inputs, labels=labels)
                loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
                a = torch.nn.CrossEntropyLoss()
                # a = outputs[1]
                # print(a.shape)
                # print(a)
                print(loss, a(outputs[1].reshape(outputs[1].shape[1], outputs[1].shape[2])[:-1], inputs.reshape(inputs.shape[1])[1:]))

                ##### Adding the dictionary entry to the loss
                # tune the hyperparameter here#####################
                key_in = tuple(inputs.detach().squeeze().cpu().numpy())
                reward = train_dict[key_in]
                print(train_dict[key_in])
                reward = pho if reward == 0 else 0
                # increase the loss
                loss = loss * (1 + reward)


                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward(retain_graph=True)

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        # Log metrics
                        if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                            results = evaluate(args, model, tokenizer)
                            for key, value in results.items():
                                tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                        tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                        logging_loss = tr_loss

                    if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                        print("Saving model...")
                        # Save model checkpoint
                        ############################# SAVE MODEL#####################################
                        # output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                        output_dir = os.path.join(args.output_dir)

                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, 'training_args.bin'))

                        ############################################################
                        # input("Press Enter to proceed to next epoch")

                if args.max_steps > 0 and global_step > args.max_steps:
                    epoch_iterator.close()
                    break
            if args.max_steps > 0 and global_step > args.max_steps:
                train_iterator.close()
                break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    print("Training Complete!!! =============")
    ############# STORE CONTEXT FILE
    curr_path = os.getcwd()
    curr_path = curr_path.replace("/ggbert_classifier", "")
    context_file = open("./RL_data/cached_dict.p", 'wb+')
    pickle.dump(context_dict, context_file)
    #########################################
    return global_step, tr_loss


def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    # args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = 2
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    # for batch in tqdm(eval_dataloader, desc="Evaluating"):
    for batch in eval_dataloader:
        batch = batch.to(args.device)

        with torch.no_grad():
            outputs = model(batch, masked_lm_labels=batch) if args.mlm else model(batch, labels=batch)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {
        "perplexity": perplexity
    }

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result

#pho is hyperparameter we need to tune
def run_finetune_gpt2(gpu_use, pho):
    parser = argparse.ArgumentParser()

    #########################set the parameter#########################################
    ## =================== make sure to locate
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_use
    import sys
    curr_path = os.getcwd()
    sys.path.append(curr_path)
    #######################Change in the first step#######################
    parser.add_argument("--train_data_file", default='./RL_data/RL_test_1st.csv', type=str,
                        required=False,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default='./gpt2_models/roc_model_RL_tune_v2', type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    #####################################################################################
    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--model_type", default="gpt2", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default= curr_path + "/", type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", default=True,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.", default=False)
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    ############## CHANGE SAVE STEP #####################
    parser.add_argument('--save_steps', type=int,
                        #default=50,
                        default=500,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir',
                        # action='store_true',
                        default=True, required=False,
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache',
                        # action='store_true',
                        default=True, required=False,
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()

    # USE THE MODEL LAST STEP GENERATE# DEFAULT WAS GPT2#######change in the second step#########
    args.model_name_or_path = './gpt2_models/roc_model_RL'
    ##################################

    if args.model_type in ["bert", "roberta"] and not args.mlm:
        raise ValueError("BERT and RoBERTa do not have LM heads but masked LM heads. They must be run using the --mlm "
                         "flag (masked language modeling).")
    if args.eval_data_file is None and args.do_eval:
        raise ValueError("Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
                         "or remove the --do_eval argument.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 4

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_use
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    ###################################################################################################################################
    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make tokensure only the first process in distributed training download model & vocab

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    print('model_class',model_class)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
    print('args.device',args.device)
    model.to(args.device)
    # for name, param in model.transformer.state_dict().items():
    #######Change in the third step###############

    #Change the frozen weight here PS: False = frozen
    # for name, param in model.transformer.named_parameters():
    #     param.requires_grad = False if name not in tune_name_list else True
    #     if name == 'h.0.attn.c_attn.weight':
    #         print(name, param)


    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)
    args.do_train = True
    # Training#
    if args.do_train:
        print('DOIING TRAINNING')
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss = train(args, train_dataset, model, tokenizer, pho)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    import glob
    output_dir_gpt2 = args.output_dir

    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(output_dir_gpt2) and args.local_rank in [-1, 0]:
            os.makedirs(output_dir_gpt2)

        logger.info("Saving model checkpoint to %s", output_dir_gpt2)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir_gpt2)

        tokenizer.save_pretrained(output_dir_gpt2)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(output_dir_gpt2, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(output_dir_gpt2)
        tokenizer = tokenizer_class.from_pretrained(output_dir_gpt2, do_lower_case=args.do_lower_case)
        model.to(args.device)

        #############for checking if freeze correct part
        for name, param in model.transformer.named_parameters():
            # print(name, param.shape)
            if name == 'h.0.attn.c_attn.weight':
                print(name, param)


if __name__ == "__main__":
    run_finetune_gpt2(gpu_use='0', pho=0.001)