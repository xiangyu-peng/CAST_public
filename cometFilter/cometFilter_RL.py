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

#import comet model
import sys
import os
sys.path.append('../comet-commonsense')  # edit by yourself pls
import src.data.data as data
import src.data.config as cfg
import src.interactive.functions as interactive
from parser.parser import *
from similarity import *
from beam_search.search_sim import diverse_beam, decode_beam_search
# from beam_search.search_tfidf_ex import diverse_beam, decode_beam_search
from stanfordcorenlp import StanfordCoreNLP
from coreference_resolution import coref_filter
from finetune_gpt2 import run_finetune_gpt2
######

import argparse
import logging
import csv
import re
import os

import numpy as np
import torch
import logging


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
        sys.stdout = open(os.devnull, 'w')

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
        logger.info("CTRL typically works better with lower temperatures (and lower top_k).")

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not any(encoded_prompt[0] == x for x in tokenizer.control_codes.values()):
        logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")
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
                language = input("Using XLM. Select language in " + str(list(available_languages)) + " >>> ")

        model.config.lang_id = model.config.lang2id[language]
        # kwargs["language"] = tokenizer.lang2id[language]

    # TODO fix mask_token_id setup when configurations will be synchronized between models and tokenizers
    # XLM masked-language modeling (MLM) models need masked token
    # is_xlm_mlm = "mlm" in args.model_name_or_path
    # if is_xlm_mlm:
    #     kwargs["mask_token_id"] = tokenizer.mask_token_id

    return prompt_text


def prepare_xlnet_input(args, _, tokenizer, prompt_text):
    prefix = args.prefix if args.prefix else args.padding_text if args.padding_text else PREFIX
    prompt_text = prefix + prompt_text
    return prompt_text


def prepare_transfoxl_input(args, _, tokenizer, prompt_text):
    prefix = args.prefix if args.prefix else args.padding_text if args.padding_text else PREFIX
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
    while True:
        # Different models need different input formatting and/or extra arguments
        requires_preprocessing = args.model_type in PREPROCESSING_FUNCTIONS.keys()
        if requires_preprocessing:
            prepare_input = PREPROCESSING_FUNCTIONS.get(args.model_type)
            preprocessed_prompt_text = prepare_input(args, model, tokenizer, prompt_text)

            if model.__class__.__name__ in ["TransfoXLLMHeadModel"]:
                tokenizer_kwargs = {"add_space_before_punct_symbol": True}
            else:
                tokenizer_kwargs = {}

            encoded_prompt = tokenizer.encode(
                preprocessed_prompt_text, add_special_tokens=False, return_tensors="pt", **tokenizer_kwargs
            )
        else:
            prefix = args.prefix if args.prefix else args.padding_text
            encoded_prompt = tokenizer.encode(prefix + prompt_text, add_special_tokens=False, return_tensors="pt")
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
            num_return_sequences=args.search_time * 2  # args.num_return_sequences
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
            text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

            # Remove all text after the stop token
            text = text[: text.find(args.stop_token) if args.stop_token else None]

            #####bbb#####
            text = text.replace("\n", ".")
            text = text.replace("?", ".")
            text = text.replace("!", ".")
            text = text.split(".")[num_sen] + '.'  # generated sentence

            if len(tokenizer.encode(text)) < 18 and len(tokenizer.encode(text)) > 5:
            # if len(text.split(' ')) < 13 and len(text.split(' ')) > 5:
                sen_id += 1
                text_gen_dict[sen_id] = text
                if sen_id >= args.search_time * 2:
                    return text_gen_dict
        return text_gen_dict

def level_set(level='weak', type='x'):
    """
    According to the filter level to decide the match condition
    :return: category list
    """
    if level == 'weak':
        if type == 'x':
            category = ['xAttr', 'xEffect', 'xIntent', 'xNeed', 'xReact', 'xWant']
        elif type == 'o':
            category = ['oEffect', 'oReact', 'oWant']
        else:
            category = []
            print('wrong type')
        return category

    elif level == 'medium':
        if type == 'x':
            category = ['xEffect', 'xReact', 'xWant']
        elif type == 'o':
            category = ['oEffect', 'oReact', 'oWant']
        else:
            category = []
            print('wrong type')
        return category

    elif level == 'react':
        if type == 'x':
            category = ['xAttr']
        elif type == 'o':
            category = ['oReact']
        else:
            category = []
            print('wrong type')
        return category

    elif level == 'want':
        if type == 'x':
            category = ['xIntent']
        elif type == 'o':
            category = ['oWant']
        else:
            category = []
            print('wrong type')
        return category

    elif 'strong' in level:
        if type == 'x':
            category = ['xIntent', 'xNeed', 'xAttr']
        elif type == 'o':
            category = ['oWant', 'oEffect', 'oReact']
        else:
            category = []
            print('wrong type')
        return category

    elif level == 'effect':
        if type == 'x':
            category = ['xNeed']
        elif type == 'o':
            category = ['oEffect']
        else:
            category = []
            print('wrong type')
        return category

    else:
        print('Wrong level')
        return []



def comet_filter(input_event, model, sampler, data_loader, text_encoder, type='x', level='weak'):

    category = level_set(level, type)
    with HiddenPrints():
        outputs = interactive.get_atomic_sequence(
            input_event, model, sampler, data_loader, text_encoder, category=category)

    output_list = []
    for key in category:
        value = outputs[key]['beams']
        # value = [i for i in value if i != 'none']
        output_list += value

    return output_list

def check_match(input_s ,output_s, model_comet, sampler_comet, data_loader_comet, text_encoder_comet, level, beam_size, pair_dict=None):
    """
    We apply to catch the match
    :param input_s:
    :param output_s:
    :param model_comet:
    :param sampler_comet:
    :param data_loader_comet:
    :param text_encoder_comet:
    :param level:
    :param beam_size:
    :return:
    """
    output_comet_input = comet_filter(input_event=input_s,
                                model=model_comet,
                                sampler=sampler_comet,
                                data_loader=data_loader_comet,
                                text_encoder=text_encoder_comet,
                                type='o',
                                level=level)  # can put outside this func to improve effciency

    output_comet_output = comet_filter(input_event=output_s,
                                model=model_comet,
                                sampler=sampler_comet,
                                data_loader=data_loader_comet,
                                text_encoder=text_encoder_comet,
                                type='x',
                                level=level)

    if level == 'strong':
        match = 0
        for att in range(3):
            if similarity_detect(queries=output_comet_input[att*beam_size: (att+1)*beam_size],
                                 corpus=output_comet_output[att * beam_size: (att + 1) * beam_size],
                                 threshold=args.sim_threshold):
                match += 1
        if not match and pair_dict != None and 0 not in pair_dict:
            pair_dict[0] = input_s + output_s

        return True if match >= args.num_matching else False

    elif level == 'strong_1':
        for att in range(3):
            if similarity_detect(queries=output_comet_input[att*beam_size: (att+1)*beam_size],
                                 corpus=output_comet_output[att * beam_size: (att + 1) * beam_size],
                                 threshold=args.sim_threshold):
                # print(output_comet_input[att*beam_size: (att+1)*beam_size])
                # print(output_comet_output[att * beam_size: (att + 1) * beam_size])
                # print(input_s, output_s)
                return True

    elif level == 'weak' or level == 'medium' or level == 'react' or level == 'want' or level == 'effect':
        if similarity_detect(queries=output_comet_input,
                             corpus=output_comet_output,
                             threshold=args.sim_threshold):
            return True

    return False

def parser_filter(gen_dict, chars):
    output_dict = dict()
    id_new = 0
    for id in range(1, len(gen_dict)+1):
        text = gen_dict[id]
        new_chars = parse_sentence(text)
        if sorted(new_chars) == sorted(chars) or len(new_chars) <= 1:
            id_new += 1
            output_dict[id_new] = text
            if id_new >= args.search_time:
                return output_dict
    return output_dict

def backtrack(story, ):
    pass

def main():
    # Add nlp server here
    nlp = StanfordCoreNLP(r'../stanford-corenlp-4.1.0')
    # Initialize the model and tokenizer
    try:
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    except KeyError:
        raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)

    if args.fp16:
        model.half()

    args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)
    logger.info(args)

    # load COMeT model here
    if args.comet_use:
        beam_size = int(re.search('\d+', args.comet_algorithm).group())
        opt_comet, state_dict_comet = interactive.load_model_file(args.model_file)
        data_loader_comet, text_encoder_comet = interactive.load_data("atomic", opt_comet)
        n_ctx_comet = data_loader_comet.max_event + data_loader_comet.max_effect
        n_vocab_comet = len(text_encoder_comet.encoder) + n_ctx_comet
        model_comet = interactive.make_model(opt_comet, n_vocab_comet, n_ctx_comet, state_dict_comet)
        sampler_comet = interactive.set_sampler(opt_comet, args.comet_algorithm, data_loader_comet)

        # TODO: set device for COMet.
        if args.device_id != -1:
            cfg.device = int(args.device_id)
            # cfg.set_device(int(args.device_id))
            cfg.do_gpu = True
            torch.cuda.set_device(cfg.device)
            model_comet.cuda(cfg.device)
        else:
            cfg.device = "cpu"

    # in future, we will add parser to get charaters name, now I use a dict
    # record all the pairs in the dictionary
    if args.RL_use:
        pairs_dict = dict()

    matching_count = 0
    with open(args.prompt_file, encoding="utf-8") as f:
        story_id = -1  # the story id depends on # of prompts in the file
        # each line is a opening of each story
        for row in f.readlines():
            story_id += 1
            sentence_comet = row.strip()  # the sentence should have period!

            # We use NER and dependency tree to extract two chars from the input
            if args.char_name_use:
                with HiddenPrints():
                    chars = parse_sentence(sentence_comet)
                # logging info
                logger.info('===================')
                logger.info('sen => ', sentence_comet)
                logger.info('chars ===>', chars)
                logger.info('===================')
            else:
                chars = ['[MALE]', '[FEMALE]']
            if not args.two_char_use:
                chars = ['[MALE]', '[MALE]']

            story_sen_id = 0  # the first sentence's id = 0
            if args.backtrack_use:
                story_beams = dict()  # record the beams of the stories.
                story_beams[story_sen_id] = [sentence_comet] * args.backtrack_threshold  # record the first sentence into beams

            story = [sentence_comet]
            prompt_text = sentence_comet
            first_prompt = sentence_comet
            prompt_add_con_sig = True


            while story_sen_id < args.story_length - args.prompt_len:
                story_sen_id += 1

                if args.backtrack_use:
                    if story_sen_id not in story_beams:
                        story_beams[story_sen_id] = []

                if prompt_add_con_sig:  # only add the condition when needed.
                    prompt_text += ' ' + chars[story_sen_id % 2]

                # logger.info('prompt_text' + str(prompt_text))
                stop_sig = True
                pair_dict = dict() if args.RL_use else None  # define the pair_dict we need to record the sentence
                while stop_sig:
                    if args.beam_search:
                        exclude_chars = chars.copy()
                        if not args.exclude_chars:  # model, length, raw_text, neural=False, num_samples=3, lamb=0.0005, temperature=1, exclude_words=[]
                            exclude_chars = []

                        text_gen_dict = decode_beam_search(diverse_beam(model=model,
                                                                        length=14,
                                                                        raw_text=prompt_text,
                                                                        num_samples=int(args.search_time * 1.5),
                                                                        lamb=args.beam_lamb,
                                                                        exclude_words=exclude_chars,
                                                                        neural=args.glove_use), int(args.search_time * 1.5), story_sen_id)

                    else:
                        text_gen_dict = gen_text(prompt_text, model, tokenizer, story_sen_id)

                    if args.filter_parser:
                        with HiddenPrints():
                            text_gen_dict = parser_filter(text_gen_dict, chars)

                    if args.coref_use:
                        text_gen_dict = coref_filter(gen_dict=text_gen_dict,
                                                     chars=chars,
                                                     prev_text=first_prompt,
                                                     search_time=args.search_time)

                    if args.backtrack_use:
                        match_found = False
                        matching = 0
                        for id in range(1, len(text_gen_dict) + 1):
                            text = text_gen_dict[id]
                            matching_this_round = False
                            if not args.comet_use or check_match(input_s=sentence_comet,
                                                                 output_s=text,
                                                                 model_comet=model_comet,
                                                                 sampler_comet=sampler_comet,
                                                                 data_loader_comet=data_loader_comet,
                                                                 text_encoder_comet=text_encoder_comet,
                                                                 level=args.filter_level,
                                                                 beam_size=beam_size):
                                logger.info('text' + text)
                                logger.info('MATCH and stop' + str(id) + args.filter_level)

                                story_beams[story_sen_id].append(text)
                                matching += 1
                                match_found = True
                                matching_this_round = True
                                if matching > args.backtrack_threshold:
                                    break

                        stop_sig = False

                        if not match_found:
                            story_sen_id -= 1
                            logger.info('NOT FOUND!!!!!!!!!')
                        else:
                            story.append(story_beams[story_sen_id].pop(0))

                    else:


                        for level_local in [args.filter_level, 'strong_1', 'weak']:
                            for id in range(1, len(text_gen_dict) + 1):
                                text = text_gen_dict[id]
                                #
                                # print('sentence_comet', sentence_comet)
                                # print('text', text)
                                # check if this sentence satisfy the requirement
                                matching_count += 1
                                if not args.comet_use or check_match(input_s=sentence_comet,
                                                                     output_s=text,
                                                                     model_comet=model_comet,
                                                                     sampler_comet=sampler_comet,
                                                                     data_loader_comet=data_loader_comet,
                                                                     text_encoder_comet=text_encoder_comet,
                                                                     level=level_local,
                                                                     beam_size=beam_size,
                                                                     pair_dict=pair_dict):


                                    # print('matching_count + 1')
                                    if args.RL_use and 1 not in pair_dict:
                                        # print(pair_dict)
                                        pair_dict[1] = sentence_comet + text
                                    logger.info('MATCH and stop' + str(id) + level_local)
                                    print('MATCH and stop' , str(id) , level_local)
                                    stop_sig = False
                                    break

                            if id < len(text_gen_dict) or not stop_sig:
                                break

                        # print('pair_dict', pair_dict)
                        story.append(text)  # append story

                # whether to use the whole story to generate next sentence
                if args.backtrack_use and not match_found:
                    print(story_beams)
                    # did not find match, we will use the dictionary to replace the previous sentence
                    if story_beams[story_sen_id]: # if no history, we have to use the old prompt again.
                        story.pop(-1)
                        print(story_beams[story_sen_id])
                        story.append(story_beams[story_sen_id].pop(0))
                        prompt_add_con_sig = True
                        if args.history_use:
                            prompt_text = " ".join(story)
                        else:
                            prompt_text = story[-1]
                    else:
                        prompt_add_con_sig = False
                else:
                    prompt_add_con_sig = True
                    if args.history_use:
                        prompt_text = "".join(story)
                    else:
                        prompt_text = text

                    sentence_comet = text  # update the next first sentence

                # if use RL, we need to consider the dictionary to save the pairs.
                if args.RL_use:
                    # print(pair_dict)
                    pairs_dict[str(story_id) + str(story_sen_id)] = pair_dict
                    # print(pairs_dict)

            # write the story into file
            story = "".join(story)
            print('whole story:', story)
            with open(args.story_save_path, 'a') as out_file:
                txt_writer = csv.writer(out_file, delimiter='\t')
                txt_writer.writerow([story.strip()])

    # print(pairs_dict)
    # output the pairs to csv file:
    if args.RL_use:
        with open(args.RL_saving_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)

            for key in pairs_dict:
                if 0 in pairs_dict[key]:
                    writer.writerow([0] + [pairs_dict[key][0]])
                if 1 in pairs_dict[key]:
                    writer.writerow([1] + [pairs_dict[key][1]])
            # writer.writerow([matching_count / story_id / 5])
            print(story_id, matching_count , 'COUNT', matching_count / (story_id) / 5)

    # turn off nlp
    nlp.close()

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default='gpt2',
        type=str,
        required=False,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--length", type=int, default=50)
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)

    parser.add_argument("--prefix", type=str, default="", help="Text added prior to input.")
    parser.add_argument("--padding_text", type=str, default="", help="Deprecated, the use of `--prefix` is preferred.")
    parser.add_argument("--xlm_language", type=str, default="", help="Optional language when used with the XLM model.")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    #####bbb#####
    parser.add_argument("--prompt_file",
                        type=str,
                        default=None,
                        required=True,
                        help="file stored the first sentence")
    # parser.add_argument("--character_file",
    #                     type=str,
    #                     default=None,
    #                     required=False,
    #                     help="file stored the first and second characters")
    parser.add_argument("--story_length",
                        type=int,
                        default=5,
                        help="# of sentences in this story")
    parser.add_argument("--story_save_path",
                        type=str,
                        default='data_story/story_gen_',
                        help="path saving the story")
    parser.add_argument("--comet_use",
                        type=str2bool,
                        default=False,
                        help="whether to use comet filtering or not")
    parser.add_argument("--device_id",
                        type=int,
                        default=0,
                        help="gpu id")
    parser.add_argument("--history_use",
                        type=str2bool,
                        default=True,
                        help="whether to use the generated text as promp to generate next sentence")

    parser.add_argument("--prompt_len",
                        type=int,
                        default=1,
                        help="use 1 or 2 sentence in the doc as the first prompt, default is 1 sentence")
    #####COMeT#####
    parser.add_argument("--model_file",
                        type=str,
                        default="../comet-commonsense/pretrained_models/atomic_pretrained_model.pickle",
                        help="pretrain weight for COMeT")
    parser.add_argument("--comet_algorithm",
                        type=str,
                        default="beam-5",
                        help="algorithm for COMeT, beam-#; greedy, top-#")
    parser.add_argument("--search_time",
                        type=int,
                        default=100,
                        help="# of search before you choose a random sentence")
    parser.add_argument("--filter_level",
                        type=str,
                        default='weak',
                        help="weak/medium/strong/react/'want/want-feel")
    parser.add_argument("--random_tech",
                        type=str,
                        default='weak',
                        help="weak/medium/strong/React")
    parser.add_argument("--filter_parser",
                        type=str2bool,
                        default=False,
                        help="whether to filter the sentence with the 3rd characters, use parser!")
    parser.add_argument("--sim_threshold",
                        type=float,
                        default=0.8,
                        help="the threshold we use in similarity_check()")
    parser.add_argument("--num_matching",
                        type=int,
                        default=1,
                        help="indicating how many matching has to be reached")
    #####################
    #####beam search#####
    parser.add_argument("--beam_search",
                        type=str,
                        default=None,
                        help="hamming")
    parser.add_argument("--beam_lamb",
                        type=float,
                        default=0.0,
                        help="how much you wanna penalize the repeated words")
    parser.add_argument("--exclude_chars",
                        type=str2bool,
                        default=True,
                        help="how much you wanna penalize the repeated words")
    parser.add_argument("--glove_use",
                        type=str2bool,
                        default=False,
                        help="whether to use glove embedding in diverse beam")
    #####################
    ###coref resolution##
    parser.add_argument("--coref_use",
                        type=str2bool,
                        default=False,
                        help="whether to use coref to filter out the sentences.")
    #####################
    ###no_char_name##
    parser.add_argument("--char_name_use",
                        type=str2bool,
                        default=True,
                        help="whether to use coref to filter out the sentences.")
    #####################
    ###backtrack##
    parser.add_argument("--backtrack_use",
                        type=str2bool,
                        default=False,
                        help="whether to use backtrack when we cannot find a match within the threshold.")
    parser.add_argument("--backtrack_threshold",
                        type=int,
                        default=10,
                        help="number of sentences saved in backtrack dict")
    ### single character or multiple characters?#####
    parser.add_argument("--two_char_use",
                        type=str2bool,
                        default=True,
                        help="if we only use one char, we will set this as False, otherwise it is True")
    ### Use RL or not#####
    parser.add_argument("--RL_use",
                        type=str2bool,
                        default=False,
                        help="whether to use RL to do the fine-tuning")
    parser.add_argument("--RL_saving_file",
                        type=str,
                        default='./RL_data/RL_test.csv',
                        help="whether to use backtrack when we cannot find a match within the threshold.")
    #####################
    args = parser.parse_args()
    args.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    ###define the story text file name###
    if args.comet_use:
        args.story_save_path += 'fl_' + args.filter_level
    else:
        args.story_save_path += 'baseline'

    if args.beam_search:
        args.story_save_path += '_bs_' + args.beam_search + '_' + str(args.beam_lamb)

    if args.exclude_chars:
        args.story_save_path += '_ex_chars'

    if args.coref_use:
        args.story_save_path += '_coref'

    if args.history_use:
        args.story_save_path += '_history'

    if not args.char_name_use:
        args.story_save_path += '_noname'

    if args.backtrack_use:
        args.story_save_path += '_bt'

    if not args.two_char_use:
        args.story_save_path += '_1char'

    if args.num_matching != 1:
        args.story_save_path += '_numMat' + str(args.num_matching)

    args.story_save_path += '.txt'
    ######################################

    logger.warning(
        "device: %s, n_gpu: %s, 16-bits training: %s",
        args.device,
        args.n_gpu,
        args.fp16,
    )

    set_seed(args.seed)
    main()
    # for _ in range(1):
    #     if os.path.exists(args.RL_saving_file):
    #         os.remove(args.RL_saving_file)
    #     for seed in range(2, 20):
    #         set_seed(seed)
    #         main()
    #     # run_finetune_gpt2(gpu_use='1', pho=0.001)


