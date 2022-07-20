#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""

# import comet model
import sys
import os

# sys.path.append(
#     "../comet-commonsense"
# )  # edit by yourself pls
# import src.data.data as data
# import src.data.config as cfg
# import src.interactive.functions as interactive
from parser.parser import *
from similarity import *

# from beam_search.search_tfidf_ex import diverse_beam, decode_beam_search
from stanfordcorenlp import StanfordCoreNLP
from coreference_resolution import coref_filter
from comet_2020_use import comet_2020_Filter, load_model

######

import argparse
import logging
import csv
import re
from tqdm import tqdm, trange

import numpy as np
import torch

from nltk_gen import nltk_gen

from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
)
from boost import Boost_Prob

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.CRITICAL,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PREFIX = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


def prepare_ctrl_input(args, _, tokenizer, prompt_text):
    if args.temperature > 0.7:
        logger.info(
            "CTRL typically works better with lower temperatures (and lower top_k)."
        )

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not any(encoded_prompt[0] == x for x in tokenizer.control_codes.values()):
        logger.info(
            "WARNING! You are not starting your generation from a control code so you won't get good results"
        )
    return prompt_text


def prepare_xlm_input(args, model, tokenizer, prompt_text):
    # kwargs = {"language": None, "mask_token_id": None}

    # Set the language
    use_lang_emb = hasattr(model.config, "use_lang_emb") and model.config.use_lang_emb
    if hasattr(model.config, "lang2id") and use_lang_emb:
        available_languages = model.config.lang2id.keys()
        if args.xlm_language in available_languages:
            language = args.xlm_language
        else:
            language = None
            while language not in available_languages:
                language = input(
                    "Using XLM. Select language in "
                    + str(list(available_languages))
                    + " >>> "
                )

        model.config.lang_id = model.config.lang2id[language]
        # kwargs["language"] = tokenizer.lang2id[language]

    # TODO fix mask_token_id setup when configurations will be synchronized between models and tokenizers
    # XLM masked-language modeling (MLM) models need masked token
    # is_xlm_mlm = "mlm" in args.model_name_or_path
    # if is_xlm_mlm:
    #     kwargs["mask_token_id"] = tokenizer.mask_token_id

    return prompt_text


def prepare_xlnet_input(args, _, tokenizer, prompt_text):
    prefix = (
        args.prefix
        if args.prefix
        else args.padding_text
        if args.padding_text
        else PREFIX
    )
    prompt_text = prefix + prompt_text
    return prompt_text


def prepare_transfoxl_input(args, _, tokenizer, prompt_text):
    prefix = (
        args.prefix
        if args.prefix
        else args.padding_text
        if args.padding_text
        else PREFIX
    )
    prompt_text = prefix + prompt_text
    return prompt_text


PREPROCESSING_FUNCTIONS = {
    "ctrl": prepare_ctrl_input,
    "xlm": prepare_xlm_input,
    "xlnet": prepare_xlnet_input,
    "transfo-xl": prepare_transfoxl_input,
}


def gen_text_decoding(prompt_text, model, tokenizer, num_sen, boost_obj, features, story_sen_id, args, topk, device):
    while True:
        # Different models need different input formatting and/or extra arguments
        requires_preprocessing = args.model_type in PREPROCESSING_FUNCTIONS.keys()
        if requires_preprocessing:
            prepare_input = PREPROCESSING_FUNCTIONS.get(args.model_type)
            preprocessed_prompt_text = prepare_input(
                args, model, tokenizer, prompt_text
            )

            if model.__class__.__name__ in ["TransfoXLLMHeadModel"]:
                tokenizer_kwargs = {"add_space_before_punct_symbol": True}
            else:
                tokenizer_kwargs = {}

            encoded_prompt = tokenizer.encode(
                preprocessed_prompt_text,
                add_special_tokens=False,
                return_tensors="pt",
                **tokenizer_kwargs
            )
        else:
            prefix = args.prefix if args.prefix else args.padding_text
            encoded_prompt = tokenizer.encode(
                prefix + prompt_text, add_special_tokens=False, return_tensors="pt"
            )

        encoded_prompt = encoded_prompt.to(device)

        if encoded_prompt.size()[-1] == 0:
            input_ids = None
        else:
            input_ids = encoded_prompt

        # generate token by token
        # output_sequences = model.generate(
        #     input_ids=input_ids,
        #     max_length=args.length + len(encoded_prompt[0]),
        #     temperature=args.temperature,
        #     top_k=args.k,
        #     top_p=args.p,
        #     repetition_penalty=args.repetition_penalty,
        #     do_sample=True,
        #     num_return_sequences=args.search_time * 2  # args.num_return_sequences
        # )
        text_gen_dict = dict()
        text_gen_dict_no_boost = dict()
        id = 1
        for i in range(1, num_sen*20):
            text_gen, done = gen_token_by_token(
                input_ids,
                temperature=1,
                model=model,
                boost_obj=boost_obj,
                features=features,
                tokenizer=tokenizer,
                topk=topk,
                device=device,
            )
            # print('--->', text_gen, text_gen.split('.'), story_sen_id, done)
            if i <= num_sen:
                text_gen_dict_no_boost[i] = text_gen.split('.')[story_sen_id] + '.'
            if done:
                text_gen_dict[id] = text_gen.split('.')[story_sen_id] + '.'
                id += 1
                if id > num_sen:
                    break

        # append text w/o boosting into return dict
        id_start = len(text_gen_dict) + 1
        for key in text_gen_dict_no_boost:
            if len(text_gen_dict_no_boost[key].split()) > 5:
                text_gen_dict[id_start] = text_gen_dict_no_boost[key]

        return text_gen_dict


def choose_from_top(probs, n):
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob)  # Normalize
    choice = np.random.choice(n, 1, p=top_prob)
    token_id = ind[choice][0]
    return int(token_id)


def gen_token_by_token(cur_token_ids, temperature, model, boost_obj, features, tokenizer, device, topk=5):
    previous_id = 0
    done_sig = 0
    for i in range(100):
        # print('cur_token_ids', cur_token_ids)
        outputs = model(input_ids=cur_token_ids)
        # print('outputs',outputs)
        logits = outputs[0]
        next_word_logits = logits[0, -1] / temperature
        softmax_logits = torch.softmax(next_word_logits, dim=0)

        probs = softmax_logits.to("cpu").detach().numpy()

        # boost the prob based on comet features
        probs, done = boost_obj.boost(tokenizer, probs, features, previous_id=previous_id)
        next_token_id = choose_from_top(probs, n=topk)

        previous_id = next_token_id

        # Stop when end word is predicted
        if next_token_id == 50256 or i >= 20:  # Token for <|endoftext|>
            break
        done_sig = max(done, done_sig)
        # append to cur tokens
        next_token_id = torch.tensor([[next_token_id]]).to(device)
        cur_token_ids = torch.cat((cur_token_ids, next_token_id), dim=-1)

    output_text = tokenizer.decode(list(cur_token_ids.to("cpu").detach().numpy())[0])
    # print(".".join(output_text.split(".")[0:2]))
    # print('output_text', output_text)
    return output_text, done_sig


def features_dealt(list_of_list, chars_names, prompt_text=None):
    features_one = []
    for f in list_of_list:
        features_one += f
    nltk_model = nltk_gen()
    features_nltk = []
    # print('features_one', features_one)
    for f in features_one:
        after_fs = nltk_model.comet_to_delete_all_possible(words=f)
        for after_f in after_fs:
            if after_f:
                after_f = (
                    after_f.replace("PersonX", chars_names[0])
                    .replace("PersonY", chars_names[1])
                    .replace("personx", chars_names[0])
                    .replace("persony", chars_names[1])
                    # .replace("Char", chars_names[0])
                    # .replace("char", chars_names[0])
                )
                if 'Char' in after_f and '[Char' not in after_f:
                    after_f = after_f.replace("Char", chars_names[0])
                features_nltk.append(after_f)
    return features_nltk


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--model_type",
#         default="gpt2",
#         type=str,
#         required=False,
#         help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
#     )
#     parser.add_argument(
#         "--model_name_or_path",
#         default="finetune_language_model/finetuned_model/roc_pure",
#         type=str,
#         help="Path to pre-trained model or shortcut name selected in the list: "
#         + ", ".join(MODEL_CLASSES.keys()),
#     )
#
#     parser.add_argument("--prompt", type=str, default="")
#     parser.add_argument("--length", type=int, default=50)
#     parser.add_argument(
#         "--stop_token",
#         type=str,
#         default=None,
#         help="Token at which text generation is stopped",
#     )
#
#     parser.add_argument(
#         "--temperature",
#         type=float,
#         default=1.0,
#         help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
#     )
#     parser.add_argument(
#         "--repetition_penalty",
#         type=float,
#         default=1.0,
#         help="primarily useful for CTRL model; in that case, use 1.2",
#     )
#     parser.add_argument("--k", type=int, default=0)
#     parser.add_argument("--p", type=float, default=0.9)
#
#     parser.add_argument(
#         "--prefix", type=str, default="", help="Text added prior to input."
#     )
#     parser.add_argument(
#         "--padding_text",
#         type=str,
#         default="",
#         help="Deprecated, the use of `--prefix` is preferred.",
#     )
#     parser.add_argument(
#         "--xlm_language",
#         type=str,
#         default="",
#         help="Optional language when used with the XLM model.",
#     )
#
#     parser.add_argument(
#         "--seed", type=int, default=42, help="random seed for initialization"
#     )
#     parser.add_argument(
#         "--no_cuda", action="store_true", help="Avoid using CUDA when available"
#     )
#     parser.add_argument(
#         "--num_return_sequences",
#         type=int,
#         default=1,
#         help="The number of samples to generate.",
#     )
#     parser.add_argument(
#         "--fp16",
#         action="store_true",
#         help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
#     )
#     #####bbb#####
#     parser.add_argument(
#         "--prompt_file", type=str, default=None, help="file stored the first sentence"
#     )
#     # parser.add_argument("--character_file",
#     #                     type=str,
#     #                     default=None,
#     #                     required=False,
#     #                     help="file stored the first and second characters")
#     parser.add_argument(
#         "-l", "--story_length", type=int, default=5, help="# of sentences in this story"
#     )
#     parser.add_argument(
#         "--story_save_path",
#         type=str,
#         default="data_story/story_gen_",
#         help="path saving the story",
#     )
#     parser.add_argument(
#         "--comet_use", action="store_true", help="whether to use comet filtering or not"
#     )
#     parser.add_argument("-d", "--device_id", type=int, default=0, help="gpu id")
#     parser.add_argument(
#         "--history_use",
#         action="store_true",
#         help="whether to use the generated text as promp to generate next sentence",
#     )
#     parser.add_argument(
#         "--history_len",
#         type=int,
#         default=5,
#         help="how many generated sentences are used as promp to generate next sentence",
#     )
#
#     parser.add_argument(
#         "--prompt_len",
#         type=int,
#         default=1,
#         help="use 1 or 2 sentence in the doc as the first prompt, default is 1 sentence",
#     )
#     #####COMeT#####
#     parser.add_argument(
#         "--model_file",
#         type=str,
#         default="../comet-commonsense/pretrained_models/atomic_pretrained_model.pickle",
#         help="pretrain weight for COMeT",
#     )
#     parser.add_argument(
#         "--comet_algorithm",
#         type=str,
#         default="beam-5",
#         help="algorithm for COMeT, beam-#; greedy, top-#",
#     )
#     parser.add_argument(
#         "--search_time",
#         type=int,
#         default=50,
#         help="# of search before you choose a random sentence",
#     )
#     parser.add_argument(
#         "-f",
#         "--filter_level",
#         type=str,
#         default="weak",
#         help="weak/medium/strong/react/'want/want-feel",
#     )
#     parser.add_argument(
#         "--random_tech", type=str, default="weak", help="weak/medium/strong/React"
#     )
#     parser.add_argument(
#         "--filter_parser",
#         action="store_true",
#         help="whether to filter the sentence with the 3rd characters, use parser!",
#     )
#     parser.add_argument(
#         "--sim_threshold",
#         type=float,
#         default=0.8,
#         help="the threshold we use in similarity_check()",
#     )
#     parser.add_argument(
#         "-n",
#         "--num_matching",
#         type=int,
#         default=3,
#         help="indicating how many matching has to be reached",
#     )
#     #####################
#     #####beam search#####
#     parser.add_argument("--beam_search", type=str, default=None, help="hamming")
#     parser.add_argument(
#         "--beam_lamb",
#         type=float,
#         default=0.0,
#         help="how much you wanna penalize the repeated words",
#     )
#     parser.add_argument(
#         "--exclude_chars",
#         action="store_true",
#         help="how much you wanna penalize the repeated words",
#     )
#     parser.add_argument(
#         "--glove_use",
#         action="store_true",
#         help="whether to use glove embedding in diverse beam",
#     )
#     #####################
#     ###coref resolution##
#     parser.add_argument(
#         "--coref_use",
#         action="store_true",
#         help="whether to use coref to filter out the sentences.",
#     )
#     #####################
#     ###no_char_name##
#     parser.add_argument(
#         "--char_name_use",
#         action="store_true",
#         help="whether to use coref to filter out the sentences.",
#     )
#     #####################
#     ###backtrack##
#     parser.add_argument(
#         "--backtrack_use",
#         action="store_true",
#         help="whether to use backtrack when we cannot find a match within the threshold.",
#     )
#     parser.add_argument(
#         "--backtrack_threshold",
#         type=int,
#         default=10,
#         help="number of sentences saved in backtrack dict",
#     )
#     ### single character or multiple characters?#####
#     parser.add_argument(
#         "-t",
#         "--two_char_use",
#         action="store_true",
#         help="if we only use one char, we will set this as False, otherwise it is True",
#     )
#     #### add name to files################
#     parser.add_argument(
#         "--add_name", type=str, default=None, help="add name when use other baseline"
#     )
#     #####################
#     ######new comet2020####
#     parser.add_argument(
#         "--char_type", type=str, default="s", help="s or m; (single or multiple)"
#     )
#     parser.add_argument(
#         "--comet_checkpoints",
#         type=str,
#         default="../comet-atomic-2020/models/comet_atomic2020_bart/comet-atomic_2020_BART",
#     )
#     ########################
#     ######new prompt way####
#     parser.add_argument("--use_cond", action="store_true", help="new prompt way")
#     ########################
#     args = parser.parse_args()
#     args.device = torch.device(
#         "cuda:" + str(args.device_id)
#         if torch.cuda.is_available() and not args.no_cuda
#         else "cpu"
#     )
#     args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
#
#     try:
#         args.model_type = args.model_type.lower()
#         model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
#     except KeyError:
#         raise KeyError(
#             "the model {} you specified is not supported. You are welcome to add it and open a PR :)"
#         )
#
#     tokenizer = tokenizer_class.from_pretrained("gpt2")
#     model = model_class.from_pretrained(args.model_name_or_path)
#     model.to(args.device)
#
#     if args.fp16:
#         model.half()
#     prompt_text = "[Char_1] loves ice cream."
#     boost_obj = Boost_Prob(topk_consider=10, penalty=50)
#     features = [['coffee'], ['go to bed']]
#     features = [
#         [
#             " to eat ice cream",
#             " to eat the ice cream",
#             " to eat ice cream.",
#             " to eat some ice cream",
#             " to buy ice cream",
#             " to go to the store",
#             " to buy ice cream.",
#             " eat ice cream",
#             " to have ice cream",
#             " to eat it",
#         ],
#         ["Char loves ice cream."],
#         [
#             " eats ice cream",
#             " eat ice cream",
#             " buys ice cream",
#             " eats icecream",
#             " gets a cold",
#             " gets ice cream",
#             " gets fat",
#             " gets sick",
#             " eats it",
#             " none",
#         ],
#         [
#             " ice cream",
#             " a craving",
#             " icecream",
#             " like eating",
#             " none",
#             " hungry",
#             " satisfied",
#             " happy",
#             " craving",
#             " chocolate",
#         ],
#         [
#             " PersonX eats the ice cream",
#             " PersonY buys the ice cream",
#             " PersonX buys the ice cream",
#             " Char eats the ice cream.",
#             " Char eats ice cream every day",
#             " Char eats the ice cream",
#             " Char eats ice cream.",
#             " PersonX eats ice cream",
#             " Char eats ice cream",
#             " ice cream",
#         ],
#         [
#             " ice cream",
#             " icecream",
#             " candy bar",
#             " candy store",
#             " freezer",
#             " refrigerator",
#             " dessert",
#             " food",
#             " jar",
#             " candy",
#         ],
#     ]
#     features_nltk = features_dealt(features, chars_names=["[Char_1]", "[Char_1]"], prompt_text=prompt_text)
#     # print("features_nltk", features_nltk)
#     prompt_text = "* [Char_1] * " + prompt_text
#     text_gen_dict = gen_text_decoding(
#         prompt_text=prompt_text,
#         model=model,
#         tokenizer=tokenizer,
#         num_sen=5,
#         boost_obj=boost_obj,
#         features=features_nltk,
#         args=args,
#         story_sen_id=1,
#         topk=5
#     )
#     boost_obj.next_round_clean()
#     # print("text_gen_dict", text_gen_dict)
#
#
#
#
#     # print(tokenizer.encode('ice'))
#     # print(tokenizer.encode(' ice'))
#     # print(tokenizer.encode('ice cream'))
#     # print(tokenizer.encode(' ice cream'))
#     # print(tokenizer.encode('cream'))
#     # print(tokenizer.encode(' cream'))
#     # [501]
#     # [4771]
#     # [501, 8566]
#     # [4771, 8566]
#     # [36277]
#     # [8566]
