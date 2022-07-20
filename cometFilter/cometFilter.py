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

sys.path.append(
    "../comet-commonsense"
)  # edit by yourself pls
# import src.data.data as data
# import src.data.config as cfg
# import src.interactive.functions as interactive
from parser.parser import *
from similarity import *
# from beam_search.search_sim import diverse_beam, decode_beam_search

# from beam_search.search_tfidf_ex import diverse_beam, decode_beam_search
from stanfordcorenlp import StanfordCoreNLP
from coreference_resolution import coref_filter
from comet_2020_use import comet_2020_Filter, load_model
from decoder import features_dealt, gen_text_decoding

######

import argparse
import logging
import csv
import re
from tqdm import tqdm, trange

import numpy as np
import torch

from boost import Boost_Prob

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


class HiddenPrints:
    def __enter__(self):

        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


#
# Functions to prepare models' input
#


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


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def gen_text(prompt_text, model, tokenizer, num_sen):
    # print('GGG, ', prompt_text, num_sen)
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
        encoded_prompt = encoded_prompt.to(args.device)

        if encoded_prompt.size()[-1] == 0:
            input_ids = None
        else:
            input_ids = encoded_prompt

        output_sequences = model.generate(
            input_ids=input_ids,
            max_length=args.length + len(encoded_prompt[0]),
            temperature=args.temperature,
            top_k=args.k,
            top_p=args.p,
            repetition_penalty=args.repetition_penalty,
            do_sample=True,
            num_return_sequences=args.search_time * 2,  # args.num_return_sequences
        )

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        text_gen_dict = dict()
        sen_id = 0
        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):

            # print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
            generated_sequence = generated_sequence.tolist()

            # Decode text
            text = tokenizer.decode(
                generated_sequence, clean_up_tokenization_spaces=True
            )

            # Remove all text after the stop token
            text = text[: text.find(args.stop_token) if args.stop_token else None]

            #####bbb#####
            text = text.replace("\n", ".")
            text = text.replace("?", ".")
            text = text.replace("!", ".")
            text = text.replace(";", ".")
            # print('tttttt', text)

            if args.comet_use:
                # print(text, num_sen)
                text = text.split(".")[num_sen] + "."
                # text = text.split(".")[min(num_sen, args.history_len)] + '.'  # generated sentence
            else:
                text = text.split(".")[num_sen] + "."  # generated sentence
            if len(tokenizer.encode(text)) < 180 and len(tokenizer.encode(text)) > 5:
                # if len(text.split(' ')) < 13 and len(text.split(' ')) > 5:
                sen_id += 1
                text_gen_dict[sen_id] = text
                if sen_id >= args.search_time * 2:
                    return text_gen_dict
        return text_gen_dict


def level_set(level="weak", type="x"):
    """
    According to the filter level to decide the match condition
    :return: category list
    """
    if level == "weak":
        if type == "x":
            category = ["xAttr", "xEffect", "xIntent", "xNeed", "xReact", "xWant"]
        elif type == "o":
            category = ["oEffect", "oReact", "oWant"]
        else:
            category = []
            print("wrong type")
        return category

    elif level == "medium":
        if type == "x":
            category = ["xEffect", "xReact", "xWant"]
        elif type == "o":
            category = ["oEffect", "oReact", "oWant"]
        else:
            category = []
            print("wrong type")
        return category

    elif level == "react":
        if type == "x":
            category = ["xAttr"]
        elif type == "o":
            category = ["oReact"]
        else:
            category = []
            print("wrong type")
        return category

    elif level == "want":
        if type == "x":
            category = ["xIntent"]
        elif type == "o":
            category = ["oWant"]
        else:
            category = []
            print("wrong type")
        return category

    elif "strong" in level:
        # new matching for inlg
        if "new" in level:
            if type == "self_prev":
                category = [
                    "xWant",
                    "sentence",
                    "xEffect",
                    "CausesDesire",
                    "isBefore",
                    "AtLocation",
                ]
            elif type == "self_later":
                category = [
                    "xIntent",
                    "xNeed",
                    "sentence",
                    "Desires",
                    "isAfter",
                    "AtLocation",
                ]
            elif type == "x":
                category = ["xAttr", "xIntent", "sentence"]
            elif type == "o":
                category = ["oReact", "oWant", "oEffect"]
            else:
                category = []
                print("wrong type")
        else:
            if type == "x":
                category = ["xIntent", "xNeed", "xAttr"]
                # category = ['xWant', 'xEffect', 'xReact']
            elif type == "o":
                category = ["oWant", "oEffect", "oReact"]
            else:
                category = []
                print("wrong type")
        return category

    elif level == "effect":
        if type == "x":
            category = ["xNeed"]
        elif type == "o":
            category = ["oEffect"]
        else:
            category = []
            print("wrong type")
        return category

    else:
        print("Wrong level")
        return []


# def comet_filter(input_event, model, sampler, data_loader, text_encoder, type='x', level='weak'):
def comet_filter(input_event, model, type, level, char_type, num_generate):
    category = level_set(level, type)
    with HiddenPrints():
        outputs = comet_2020_Filter(
            comet=model,
            sentence=input_event,
            type="prompt",
            num_generate=num_generate,
            char_type=char_type,
        )
    output_list = []
    # print(outputs)
    for key in category:
        # print("key", key, outputs[key])
        value = outputs[key]["beams"]
        output_list += value

    return output_list


# def check_match(input_s ,output_s, model_comet, sampler_comet, data_loader_comet, text_encoder_comet, level, beam_size):
def check_match(input_s, output_s, model_comet, level, beam_size=10, char_type="s"):

    """
    We apply to catch the match
    :param input_s: str
    :param output_s: str
    :param model_comet:
    :param sampler_comet:
    :param data_loader_comet:
    :param text_encoder_comet:
    :param beam_size: 10, int.
    :param char_type: 's' or 'm'
    :return:
    """
    # output_comet_input = comet_filter(input_event=input_s,
    #                             model=model_comet,
    #                             sampler=sampler_comet,
    #                             data_loader=data_loader_comet,
    #                             text_encoder=text_encoder_comet,
    #                             type='o',
    #                             level=level)  # can put outside this func to improve effciency

    # new comet 2020
    output_comet_input = comet_2020_Filter(
        comet=model_comet,
        sentence=input_s,
        type="prompt",
        num_generate=beam_size,
        char_type=char_type,
    )
    # if args.two_char_use:
    #     output_comet_input = comet_filter(input_event=input_s,
    #                                       model=model_comet,
    #                                       type='o',
    #                                       level=level,
    #                                       char_type=char_type,
    #                                       num_generate=beam_size)
    #
    #     output_comet_output = comet_filter(input_event=output_s,
    #                                        model=model_comet,
    #                                        type='x',
    #                                        level=level,
    #                                        char_type=char_type,
    #                                        num_generate=beam_size)
    # else:
    #     output_comet_input = comet_filter(input_event=input_s,
    #                                       model=model_comet,
    #                                       type='self_prev',
    #                                       level=level,
    #                                       char_type=char_type,
    #                                       num_generate=beam_size)
    #
    #     output_comet_output = comet_filter(input_event=output_s,
    #                                        model=model_comet,
    #                                        type='self_later',
    #                                        level=level,
    #                                        char_type=char_type,
    #                                        num_generate=beam_size)
    output_comet_output = comet_2020_Filter(
        comet=model_comet,
        sentence=output_s,
        type="continuation",
        num_generate=beam_size,
        char_type=char_type,
    )
    # print('output_comet_input', output_comet_input)
    # print('output_comet_output', output_comet_output)
    # output_comet_output = comet_filter(input_event=output_s,
    #                             model=model_comet,
    #                             sampler=sampler_comet,
    #                             data_loader=data_loader_comet,
    #                             text_encoder=text_encoder_comet,
    #                             type='x',
    #                             level=level)

    if level == "strong" or level == "strong_new":
        match = 0
        for att in range(len(output_comet_input)):
            # if similarity_detect(queries=output_comet_input[att*beam_size: (att+1)*beam_size],
            #                      corpus=output_comet_output[att * beam_size: (att + 1) * beam_size],
            #                      threshold=args.sim_threshold):
            if similarity_detect(
                queries=output_comet_input[att],
                corpus=output_comet_output[att],
                threshold=args.sim_threshold,
            ):
                match += 1
                print(output_comet_input[att])
                print(output_comet_output[att])
                # print(input_s, output_s)
        return True if match >= args.num_matching else False

    elif level == "rl":
        match = 0
        for att in range(len(output_comet_input)):
            if similarity_detect(
                queries=output_comet_input[att],
                corpus=output_comet_output[att],
                threshold=args.sim_threshold,
            ):
                match += 1
        if match > 1:
            return False
        else:
            return True

    elif level == "strong_1":
        for att in range(len(output_comet_input)):
            if similarity_detect(
                queries=output_comet_input[att],
                corpus=output_comet_output[att],
                threshold=args.sim_threshold,
            ):
                return True

    elif (
        level == "weak"
        or level == "medium"
        or level == "react"
        or level == "want"
        or level == "effect"
    ):
        if similarity_detect(
            queries=output_comet_input,
            corpus=output_comet_output,
            threshold=args.sim_threshold,
        ):
            return True

    return False


def parser_filter(gen_dict, chars):
    output_dict = dict()
    id_new = 0
    for id in range(1, len(gen_dict) + 1):
        text = gen_dict[id]
        new_chars = parse_sentence(text)
        if sorted(new_chars) == sorted(chars) or len(new_chars) <= 1:
            id_new += 1
            output_dict[id_new] = text
            if id_new >= args.search_time:
                return output_dict
    return output_dict


def main():
    file_write = open(args.save_file, 'w')
    # Add nlp server here ---> java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 30000
    nlp = StanfordCoreNLP(r"./stanford-corenlp-4.1.0")
    # Initialize the model and tokenizer
    try:
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    except KeyError:
        raise KeyError(
            "the model {} you specified is not supported. You are welcome to add it and open a PR :)"
        )

    tokenizer = tokenizer_class.from_pretrained("gpt2")
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)

    if args.fp16:
        model.half()

    args.length = adjust_length_to_model(
        args.length, max_sequence_length=model.config.max_position_embeddings
    )
    logger.info(args)

    # load COMeT model here
    if args.comet_use:
        comet = load_model(args.comet_checkpoints)  # new atomic comet 2020
        comet.model.zero_grad()
        # old comet
        # beam_size = int(re.search('\d+', args.comet_algorithm).group())
        # opt_comet, state_dict_comet = interactive.load_model_file(args.model_file)
        # data_loader_comet, text_encoder_comet = interactive.load_data("atomic", opt_comet)
        # n_ctx_comet = data_loader_comet.max_event + data_loader_comet.max_effect
        # n_vocab_comet = len(text_encoder_comet.encoder) + n_ctx_comet
        # model_comet = interactive.make_model(opt_comet, n_vocab_comet, n_ctx_comet, state_dict_comet)
        # sampler_comet = interactive.set_sampler(opt_comet, args.comet_algorithm, data_loader_comet)
        #
        # # TODO: set device for COMet.
        # if args.device_id != -1:
        #     cfg.device = int(args.device_id)
        #     # cfg.set_device(int(args.device_id))
        #     cfg.do_gpu = True
        #     torch.cuda.set_device(cfg.device)
        #     model_comet.cuda(cfg.device)
        # else:
        #     cfg.device = "cpu"

    # in future, we will add parser to get charaters name, now I use a dict

    total_matching = 0
    success_rate = 0
    count_lst = []
    with open(args.prompt_file, encoding="utf-8") as f:
        story_id = 0  # the story id depends on # of prompts in the file
        # each line is a opening of each story
        for row in tqdm(f.readlines()[args.begin_line:]):
            story_id += 1
            sentence_comet = row.strip()  # the sentence should have period!

            # We use NER and dependency tree to extract two chars from the input
            if args.char_name_use:
                with HiddenPrints():
                    chars = parse_sentence(sentence_comet)
                # print("===================")
                # print("sen => ", sentence_comet)
                # print("chars ===>", chars)
                # print("===================")
            else:
                chars = ["[Char_1]", "[Char_2]"]
            if not args.two_char_use:
                chars = ["[Char_1]", "[Char_1]"]
            if args.no_prefix:
                chars = ["", ""]

            char_type = "m" if args.two_char_use else "s"

            story_sen_id = 0  # the first sentence's id = 0

            if args.backtrack_use:
                story_beams = dict()  # record the beams of the stories.
                story_beams[story_sen_id] = [
                    sentence_comet
                ] * args.backtrack_threshold  # record the first sentence into beams

            story = [sentence_comet]
            prompt_text = sentence_comet
            first_prompt = sentence_comet
            prompt_add_con_sig = True

            while story_sen_id < args.story_length - args.prompt_len:
                story_sen_id += 1
                if args.backtrack_use:
                    if story_sen_id not in story_beams:
                        story_beams[story_sen_id] = []

                # add char T
                if prompt_add_con_sig:  # only add the condition when needed.
                    if args.use_cond:  # *T* sentence....
                        prompt_text = (
                            "* " + chars[story_sen_id % 2] + " * " + prompt_text
                        )
                    else:  # sentence.... + T
                        prompt_text += " " + chars[story_sen_id % 2]
                # print('prompt_text', prompt_text, story_sen_id)
                stop_sig = True
                if args.save_rl:
                    stop_sig_lst = [True, True]
                while stop_sig:
                    if args.beam_search:
                        exclude_chars = chars.copy()
                        if (
                            not args.exclude_chars
                        ):  # model, length, raw_text, neural=False, num_samples=3, lamb=0.0005, temperature=1, exclude_words=[]
                            exclude_chars = []

                        text_gen_dict = decode_beam_search(
                            diverse_beam(
                                model=model,
                                length=14,
                                raw_text=prompt_text,
                                num_samples=int(args.search_time * 1.5),
                                lamb=args.beam_lamb,
                                exclude_words=exclude_chars,
                                neural=args.glove_use,
                            ),
                            int(args.search_time * 1.5),
                            story_sen_id,
                        )

                    elif args.use_decoder:
                        # print('prompt_text ===>', prompt_text)
                        # print('sentence_comet', sentence_comet)
                        boost_obj = Boost_Prob(
                            topk_consider=args.topk_consider_decoder,
                            penalty=args.penalty_decoder,
                        )
                        output_comet_input = comet_2020_Filter(
                            comet=comet,
                            sentence=sentence_comet,
                            type="prompt",
                            num_generate=10,
                            char_type=char_type,
                            decode=True
                        )

                        # make comet list of list as one list -> features to boost~
                        if story_sen_id % 2 == 0:
                            features = features_dealt(list_of_list=output_comet_input,
                                                      chars_names=chars,
                                                      prompt_text=sentence_comet)
                        else:
                            features = features_dealt(list_of_list=output_comet_input,
                                                      chars_names=chars[::-1],
                                                      prompt_text=sentence_comet)

                        text_gen_dict = gen_text_decoding(
                            prompt_text,
                            model,
                            tokenizer=tokenizer,
                            num_sen=args.search_time,
                            boost_obj=boost_obj,
                            features=features,
                            story_sen_id=story_sen_id,
                            args=args,
                            topk=args.topk_decode_choose,
                            device=args.device
                        )
                        # print('text_gen_dict', text_gen_dict)
                    else:
                        # print('prompt_text =>', prompt_text)
                        text_gen_dict = gen_text(
                            prompt_text, model, tokenizer, story_sen_id
                        )
                        # print('text_gen_dict=>', text_gen_dict)

                    if args.filter_parser:
                        with HiddenPrints():
                            text_gen_dict = parser_filter(text_gen_dict, chars)

                    if args.coref_use:
                        text_gen_dict = coref_filter(
                            gen_dict=text_gen_dict,
                            chars=chars,
                            prev_text=first_prompt,
                            search_time=args.search_time,
                        )

                    if args.backtrack_use:
                        match_found = False
                        matching = 0
                        for id in range(1, len(text_gen_dict) + 1):
                            text = text_gen_dict[id]
                            # print('text', text)
                            total_matching += 1
                            # (input_s, output_s, model_comet, level, beam_size, char_type)
                            if not args.comet_use and chars[story_sen_id % 2] in text:
                                print("text in baseline", text)
                                story_beams[story_sen_id].append(text)
                                matching += 1
                                match_found = True
                                break
                            elif check_match(
                                input_s=sentence_comet,
                                output_s=text,
                                model_comet=comet,
                                level=args.filter_level,
                                char_type=char_type,
                            ):
                                # print("text", text)
                                # print("MATCH and stop", id, args.filter_level)
                                story_beams[story_sen_id].append(text)
                                matching += 1
                                match_found = True
                                if matching > args.backtrack_threshold:
                                    break
                            else:
                                pass
                        stop_sig = False

                        if not match_found:
                            story_sen_id -= 1
                            print("NOT FOUND!!!!!!!!!")
                        else:
                            story.append(story_beams[story_sen_id].pop(0))

                    else:
                        ccc = 0
                        for level_local in [args.filter_level, "strong_1", "weak"]:
                            for id in range(1, len(text_gen_dict) + 1):
                                ccc+= 1
                                text = text_gen_dict[id]
                                print(ccc, '>>>', text)
                                total_matching += 1

                                # baseline
                                if not args.comet_use and (not args.two_char_use or chars[story_sen_id % 2] in text):
                                    print("text in baseline", text)
                                    stop_sig = False
                                    break

                                # cast - match - char_name in text
                                elif args.comet_use and check_match(
                                    input_s=sentence_comet,
                                    output_s=text,
                                    model_comet=comet,
                                    level=level_local,
                                    char_type=char_type,
                                ) and (not args.two_char_use or chars[story_sen_id % 2] in text):
                                    if args.diverse:
                                        if not similarity_detect(
                                                                queries=[sentence_comet],
                                                                corpus=[text],
                                                                threshold=args.diverse_threshold):
                                            # print("MATCH and stop with not so similar", id, level_local)
                                            if level_local == args.filter_level:
                                                success_rate += 1

                                            # save correct pair in file
                                            # if args.save_rl and stop_sig_lst[1]:
                                            #     print('0 *' + sentence_comet + '*' + text)
                                            #     file_write.write('0 *' + sentence_comet + '*' + text + '\n')
                                            #     file_write.flush()
                                            #     stop_sig_lst[1] = False
                                            # elif args.save_rl and not stop_sig_lst[1]:
                                            #     pass
                                            # else:
                                            stop_sig = False
                                            count_lst.append(ccc)
                                            break
                                    else:
                                        print("MATCH and stop", id, level_local)
                                        if level_local == args.filter_level:
                                            success_rate += 1
                                        stop_sig = False
                                        break

                                # save negative pair
                                # if args.comet_use and args.save_rl and stop_sig_lst[0] and \
                                #     check_match(
                                #                 input_s=sentence_comet,
                                #                 output_s=text,
                                #                 model_comet=comet,
                                #                 level='rl',  # only to find no matching sentence
                                #                 char_type=char_type) and \
                                #         (not args.two_char_use or
                                #          chars[story_sen_id % 2] in text):
                                #
                                #     file_write.write('1 *' + sentence_comet + '*' + text + '\n')
                                #     print('1 *' + sentence_comet + '*' + text)
                                #     file_write.flush()
                                #     stop_sig_lst[0] = False


                            if id < len(text_gen_dict) or not stop_sig:
                                break

                        stop_sig = False
                        story.append(text)  # append story

                # whether to use the whole story to generate next sentence
                if args.backtrack_use and not match_found:
                    # print(story_beams)
                    # did not find match, we will use the dictionary to replace the previous sentence
                    if story_beams[
                        story_sen_id
                    ]:  # if no history, we have to use the old prompt again.
                        story.pop(-1)
                        # print(story_beams[story_sen_id])
                        story.append(story_beams[story_sen_id].pop(0))
                        prompt_add_con_sig = True
                        if args.history_use:
                            if len(story) >= args.history_len and args.comet_use:
                                prompt_text = " ".join(story[-1 * args.history_len :])
                            else:
                                prompt_text = " ".join(story)
                        else:
                            prompt_text = story[-1]
                    else:
                        prompt_add_con_sig = False
                else:
                    prompt_add_con_sig = True
                    if args.history_use:
                        if len(story) >= args.history_len and args.comet_use:
                            prompt_text = " ".join(story[-1 * args.history_len :])
                        else:
                            prompt_text = " ".join(story)
                    else:
                        prompt_text = text

                    sentence_comet = text  # update the next first sentence

            # write the story into file

            story = "".join(story)
            print("whole story:", story)
            with open(args.story_save_path, "a") as out_file:
                txt_writer = csv.writer(out_file, delimiter="\t")
                txt_writer.writerow([story.strip()])
            print(total_matching / story_id / 4)
            print(success_rate / story_id / 4)

    # return generated_sequences
    # turn off nlp
    print(total_matching / 20 / 4)
    print(success_rate / 20 / 4)
    with open(args.story_save_path, "a") as out_file:
        txt_writer = csv.writer(out_file, delimiter="\t")
        txt_writer.writerow([total_matching / 20 / 4])
        txt_writer.writerow([success_rate / 20 / 4])
        txt_writer.writerow([str(count_lst)])

    nlp.close()
    file_write.close()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default="gpt2",
        type=str,
        required=False,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--length", type=int, default=50)
    parser.add_argument(
        "--stop_token",
        type=str,
        default=None,
        help="Token at which text generation is stopped",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="primarily useful for CTRL model; in that case, use 1.2",
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)

    parser.add_argument(
        "--prefix", type=str, default="", help="Text added prior to input."
    )
    parser.add_argument(
        "--padding_text",
        type=str,
        default="",
        help="Deprecated, the use of `--prefix` is preferred.",
    )
    parser.add_argument(
        "--xlm_language",
        type=str,
        default="",
        help="Optional language when used with the XLM model.",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="The number of samples to generate.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    #####bbb#####
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        required=True,
        help="file stored the first sentence",
    )
    # parser.add_argument("--character_file",
    #                     type=str,
    #                     default=None,
    #                     required=False,
    #                     help="file stored the first and second characters")
    parser.add_argument(
        "-l", "--story_length", type=int, default=5, help="# of sentences in this story"
    )
    parser.add_argument(
        "--story_save_path",
        type=str,
        default="data_story/story_gen_",
        help="path saving the story",
    )
    parser.add_argument(
        "--comet_use", action="store_true", help="whether to use comet filtering or not"
    )
    parser.add_argument("--device_id", type=int, default=0, help="gpu id")
    parser.add_argument(
        "--history_use",
        action="store_true",
        help="whether to use the generated text as promp to generate next sentence",
    )
    parser.add_argument(
        "--history_len",
        type=int,
        default=5,
        help="how many generated sentences are used as prompt to generate next sentence",
    )

    parser.add_argument(
        "--prompt_len",
        type=int,
        default=1,
        help="use 1 or 2 sentence in the doc as the first prompt, default is 1 sentence",
    )
    #####COMeT#####
    parser.add_argument(
        "--model_file",
        type=str,
        default="../comet-commonsense/pretrained_models/atomic_pretrained_model.pickle",
        help="pretrain weight for COMeT",
    )
    parser.add_argument(
        "--comet_algorithm",
        type=str,
        default="beam-5",
        help="algorithm for COMeT, beam-#; greedy, top-#",
    )
    parser.add_argument(
        "--search_time",
        type=int,
        default=25,
        help="# of search before you choose a random sentence",
    )
    parser.add_argument(
        "-f",
        "--filter_level",
        type=str,
        default="weak",
        help="weak/medium/strong/react/'want/want-feel",
    )
    parser.add_argument(
        "--random_tech", type=str, default="weak", help="weak/medium/strong/React"
    )
    parser.add_argument(
        "--filter_parser",
        action="store_true",
        help="whether to filter the sentence with the 3rd characters, use parser!",
    )
    parser.add_argument(
        "--sim_threshold",
        type=float,
        default=0.8,
        help="the threshold we use in similarity_check()",
    )
    parser.add_argument(
        "-n",
        "--num_matching",
        type=int,
        default=3,
        help="indicating how many matching has to be reached",
    )
    #####################
    #####beam search#####
    parser.add_argument("--beam_search", type=str, default=None, help="hamming")
    parser.add_argument(
        "--beam_lamb",
        type=float,
        default=0.0,
        help="how much you wanna penalize the repeated words",
    )
    parser.add_argument(
        "--exclude_chars",
        action="store_true",
        help="how much you wanna penalize the repeated words",
    )
    parser.add_argument(
        "--glove_use",
        action="store_true",
        help="whether to use glove embedding in diverse beam",
    )
    #####################
    ###coref resolution##
    parser.add_argument(
        "--coref_use",
        action="store_true",
        help="whether to use coref to filter out the sentences.",
    )
    #####################
    ###no_char_name##
    parser.add_argument(
        "--char_name_use",
        action="store_true",
        help="whether to use coref to filter out the sentences.",
    )
    #####################
    ###backtrack##
    parser.add_argument(
        "--backtrack_use",
        action="store_true",
        help="whether to use backtrack when we cannot find a match within the threshold.",
    )
    parser.add_argument(
        "--backtrack_threshold",
        type=int,
        default=10,
        help="number of sentences saved in backtrack dict",
    )
    ### single character or multiple characters?#####
    parser.add_argument(
        "-t",
        "--two_char_use",
        action="store_true",
        help="if we only use one char, we will set this as False, otherwise it is True",
    )
    #### add name to files################
    parser.add_argument(
        "--add_name", type=str, default=None, help="add name when use other baseline"
    )
    #####################
    ######new comet2020####
    parser.add_argument(
        "--char_type", type=str, default="s", help="s or m; (single or multiple)"
    )
    parser.add_argument(
        "--comet_checkpoints",
        type=str,
        default="../comet-atomic-2020/models/comet_atomic2020_bart/comet-atomic_2020_BART",
    )
    ########################
    ######new prompt way####
    parser.add_argument("--use_cond", action="store_true", help="new prompt way")
    ########################
    ######new decoder####
    parser.add_argument(
        "-d", "--use_decoder", action="store_true", help="new decoder way"
    )
    parser.add_argument(
        "--topk_consider_decoder",
        type=int,
        default=10,
        help="# of topk tokens boosted in the decoding system",
    )
    parser.add_argument(
        "--penalty_decoder",
        type=int,
        default=10,
        help="this number times prob = new prob",
    )
    parser.add_argument(
        "--topk_decode_choose",
        type=int,
        default=5,
        help="when gpt2 generate tokens, only choose forom topk tokens after bossting",
    )
    parser.add_argument(
        "--diverse",
        action='store_true',
        help="whether to filter out those too similar continuations",
    )
    parser.add_argument(
        "--diverse_threshold",
        type=float,
        default=0.85,
        help="whether to filter out those too similar continuations",
    )
    parser.add_argument(
        "--begin_line",
        type=int,
        default=0,
        help="whether to filter out those too similar continuations",
    )
    parser.add_argument(
        "--save_rl",
        action='store_true',
        help="whether to filter out those too similar continuations",
    )
    parser.add_argument(
        "--save_file",
        type=str,
        default='data_story/rl_file',
        help="whether to filter out those too similar continuations",
    )
    parser.add_argument(
        "--no_prefix",
        action='store_true',
        help="no prompt used",
    )
    parser.add_argument(
        "--rl",
        action='store_true',
        help="no prompt used",
    )
    ########################
    args = parser.parse_args()
    args.device = torch.device(
        "cuda:" + str(args.device_id)
        if torch.cuda.is_available() and not args.no_cuda
        else "cpu"
    )
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    ###define the story text file name###
    args.save_file += '_sd_' + str(args.seed) + '.txt'

    if args.comet_use:
        args.story_save_path += "fl_" + args.filter_level
    else:
        args.story_save_path += "baseline"

    if args.beam_search:
        args.story_save_path += "_bs_" + args.beam_search + "_" + str(args.beam_lamb)

    if args.exclude_chars:
        args.story_save_path += "_ex_chars"

    if args.coref_use:
        args.story_save_path += "_coref"

    if args.history_use:
        args.story_save_path += "_his_" + str(args.history_len)

    if not args.char_name_use:
        args.story_save_path += "_noname"

    if args.backtrack_use:
        args.story_save_path += "_bt"

    if not args.two_char_use:
        args.story_save_path += "_1char"
    else:
        args.story_save_path += "_2char"

    if args.num_matching != 1:
        args.story_save_path += "_numMat" + str(args.num_matching)

    if args.filter_parser:
        args.story_save_path += "_fp"

    if args.add_name:
        args.story_save_path += "_" + args.add_name

    if args.story_length:
        args.story_save_path += "_len" + str(args.story_length)

    if args.use_decoder:
        args.story_save_path += "_decoder"

    if args.diverse:
        args.story_save_path += "_div"
    if args.rl:
        args.story_save_path += "_rl"
    args.story_save_path += "_sim" + str(int(args.sim_threshold * 100))
    args.story_save_path += "_sd_" + str(args.seed) + ".txt"
    ######################################

    logger.warning(
        "device: %s, n_gpu: %s, 16-bits training: %s",
        args.device,
        args.n_gpu,
        args.fp16,
    )

    set_seed(args.seed)

    main()
    #
    # for seed in range(10, 50):
    #     set_seed(seed)
    #     main()
