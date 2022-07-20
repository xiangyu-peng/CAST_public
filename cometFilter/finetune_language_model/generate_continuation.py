from finetune_rl import run_finetune_gpt2
import argparse
import logging
import csv
import re
import math
import numpy as np
import torch
# from boost import Boost_Prob

import sys
sys.path.append('../')
sys.path.append("../../comet-atomic-2020/models/comet_atomic2020_bart")
from nltk_gen import nltk_gen
from comet_2020_use import comet_2020_Filter, load_model
from similarity import *

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
MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}

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

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

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

class GPT_2_gen(object):
    def __init__(self, args):
        # Initialize the model and tokenizer
        try:
            args.model_type = args.model_type.lower()
            self.model_class, self.tokenizer_class = MODEL_CLASSES[args.model_type]
        except KeyError:
            raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

        # set up model and tokenizer:
        self.tokenizer = self.tokenizer_class.from_pretrained('gpt2')
        self.model = self.model_class.from_pretrained(args.model_name_or_path)
        self.model.to(args.device)

        # set args
        self.args = args
        self.args.length = adjust_length_to_model(args.length, max_sequence_length=self.model.config.max_position_embeddings)
        self.nltk_model = nltk_gen()
        # self.boost_obj = Boost_Prob(topk_consider=10, penalty=100)

        # ids
        self.eos = [self.tokenizer.encode('.'), self.tokenizer.encode(' .'), self.tokenizer.encode('. ')]
        # print('self.eos', self.eos)

        #comet
        self.comet = load_model(args.comet_checkpoints)  # new atomic comet 2020
        self.comet.model.zero_grad()

    def replace_all(self, text):
        rep = {"[Char_2]": "[Char_1]", "[Char_1]": "[Char_2]"}  # define desired replacements here

        # use these three lines to do the replacement
        rep = dict((re.escape(k), v) for k, v in rep.items())
        # Python 3 renamed dict.iteritems to dict.items so use rep.items() for latest versions
        pattern = re.compile("|".join(rep.keys()))
        return pattern.sub(lambda m: rep[re.escape(m.group(0))], text)

    def check_match(self, input_s, output_s, model_comet, beam_size=5, char_type="s"):
        output_comet_input = comet_2020_Filter(
            comet=model_comet,
            sentence=input_s,
            type="prompt",
            num_generate=beam_size,
            char_type=char_type,
        )
        output_comet_output = comet_2020_Filter(
            comet=model_comet,
            sentence=output_s,
            type="continuation",
            num_generate=beam_size,
            char_type=char_type,
        )
        match = 0
        for att in range(len(output_comet_input)):
            if similarity_detect(
                queries=output_comet_input[att],
                corpus=output_comet_output[att],
                threshold=0.8,
            ):
                match += 1
        if match > 1:
            return False  # right - 0
        else:
            return True
    def syn_lst(self, syn_dict):
        '''
        dict['beach'] = {'beach',...}
        ->
        ['beach', .... ]
        :param syn_dict:
        :return:
        '''
        # print('syn_dict', syn_dict)
        res = set()
        for s_set in syn_dict.values():
            # print('s_set', s_set)
            for syn in s_set:
                # print('syn', syn)
                syn_lst = self.nltk_model.comet_to_delete_all_possible(words=syn)
                for s in syn_lst:
                    res.add(s)
        return res

    def check_done(self, list_of_syn, text):
        # print('Begin checking whether %s in text:"%s". ' % (str(list_of_syn), text))
        for s in list_of_syn:
            text_lst = text.split('.')
            for i, t in enumerate(text_lst):
                if s in t:
                    # print('Checking whether %s in text:"%s". Results-> %s' % (str(list_of_syn), text, 'True'))
                    # print('done --->', text)
                    # print('done --->', '.'.join(text_lst[:i+1]) + '.')
                    return '.'.join(text_lst[:i+1]) + '.'
        print('Checking whether %s in text:"%s". Results-> %s' % (str(list_of_syn), text, 'False'))
        return ''

    def gen_text(self, prompt_text, order_remain, prefix_char='s', char_s=0, output_obj=False, restriction=None):
        """
        Generate tokens with prompt
        :param prompt_text:
        :param order_remain: Which sentence you want to mak up.
        :return: the sentence in the order_remain order.
        """
        if char_s == 0:
            if prefix_char == 's':
                prompt_text = '* [Char_1] * ' + prompt_text + '.'
                # print('prompt_text', prompt_text)
            else:  # prefix_char == 'm'
                prompt_text = '* [Char_2] * ' + prompt_text + '.'
                # print('prompt_text', prompt_text)
        elif char_s == 1:
            if prefix_char == 's':
                prompt_text = '* [Char_2] * ' + prompt_text + '.'
                # print('prompt_text', prompt_text)
            else:  # prefix_char == 'm'
                prompt_text = '* [Char_1] * ' + prompt_text + '.'
                # print('prompt_text', prompt_text)

        else:
            prompt_text = '* [Char_1] [Char_2] * ' + prompt_text + '.'
            print('prompt_text', prompt_text)


        while True:
            # Different models need different input formatting and/or extra arguments
            requires_preprocessing = self.args.model_type in PREPROCESSING_FUNCTIONS.keys()
            if requires_preprocessing:
                prepare_input = PREPROCESSING_FUNCTIONS.get(self.args.model_type)
                preprocessed_prompt_text = prepare_input(self.args, self.model, self.tokenizer, prompt_text)

                if self.model.__class__.__name__ in ["TransfoXLLMHeadModel"]:
                    tokenizer_kwargs = {"add_space_before_punct_symbol": True}
                else:
                    tokenizer_kwargs = {}

                encoded_prompt = self.tokenizer.encode(
                    preprocessed_prompt_text, add_special_tokens=False, return_tensors="pt", **tokenizer_kwargs
                )
            else:
                prefix = self.args.prefix if self.args.prefix else self.args.padding_text
                encoded_prompt = self.tokenizer.encode(prefix + prompt_text, add_special_tokens=False, return_tensors="pt")
            encoded_prompt = encoded_prompt.to(self.args.device)

            if encoded_prompt.size()[-1] == 0:
                input_ids = None
            else:
                input_ids = encoded_prompt

            output_sequences = self.model.generate(
                input_ids=input_ids,
                max_length=self.args.length + len(encoded_prompt[0]),
                temperature=self.args.temperature,
                top_k=self.args.k,
                top_p=self.args.p,
                repetition_penalty=self.args.repetition_penalty,
                do_sample=True,
                num_return_sequences=5  # args.num_return_sequences
            )

            # Remove the batch dimension when returning multiple sequences
            if len(output_sequences.shape) > 2:
                output_sequences.squeeze_()

            for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
                # print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
                generated_sequence = generated_sequence.tolist()

                # Decode text
                text = self.tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

                # Remove all text after the stop token
                text = text[: text.find(self.args.stop_token) if self.args.stop_token else None]

                #####becky#####
                text = text.replace("\n", ".")
                text = text.replace(":", ".")
                text = text.replace("?", ".")
                text = text.replace("!", ".")

                # print(text)
                if output_obj:
                    text = text.split(".")[order_remain]
                    prompt_text_len =len(prompt_text.split('.')[order_remain])
                    text = text[prompt_text_len+1:]

                else:
                    text = text.split(".")[order_remain] + '.'  # generated sentence

                if not restriction:
                    return text

    def gen_multiple_obj(self, prompt_text, num=1):
        """
        Generate multiple obj at once.
        :param num: # of obj you need
        :return: list of str.
        """
        res = []
        order_remain = prompt_text.count('.')
        for seed in range(self.args.seed, self.args.seed + num):
            res.append(self.gen_text(prompt_text=prompt_text,
                                     order_remain=order_remain,
                                     output_obj=True))
        return res

    def calculate_prob(self, sentences=[]):
        outputs = []
        for sentence in sentences:
            inputs = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sentence))
            inputs = torch.tensor(inputs)
            # inputs = self.tokenizer.add_special_tokens_single_sentence(inputs)
            labels = inputs.clone()
            inputs = inputs.to(self.args.device)
            labels = labels.to(self.args.device)
            output = float(self.model(inputs, labels=labels)[0].cpu().detach().numpy())
            # print('The prob of ' + sentence + ' is ===> ', output)
            outputs.append(math.exp(output))
        return outputs

    # def gen_text_decoding(self, prompt_text, num_sen, features, story_sen_id, topk, strict=False):
    #     '''
    #     Gnerate text under features
    #     :param prompt_text: str, text
    #     :param num_sen: int, number of text we need
    #     :param features: list of str.
    #     :param story_sen_id:
    #     :param topk:
    #     :param strict:
    #     :return:
    #     '''
    #     count = 0
    #     text_gen_dict_res = dict()
    #     syn_lst = self.syn_lst(self.boost_obj.generate_syn(features)[0])
    #     # print('syn_lst', syn_lst)
    #     while count < 1:
    #         count += 1
    #
    #         # Different models need different input formatting and/or extra arguments
    #         requires_preprocessing = self.args.model_type in PREPROCESSING_FUNCTIONS.keys()
    #         if requires_preprocessing:
    #             prepare_input = PREPROCESSING_FUNCTIONS.get(self.args.model_type)
    #             preprocessed_prompt_text = prepare_input(
    #                 self.args, self.model, self.tokenizer, prompt_text
    #             )
    #
    #             if self.model.__class__.__name__ in ["TransfoXLLMHeadModel"]:
    #                 tokenizer_kwargs = {"add_space_before_punct_symbol": True}
    #             else:
    #                 tokenizer_kwargs = {}
    #
    #             encoded_prompt = self.tokenizer.encode(
    #                 preprocessed_prompt_text,
    #                 add_special_tokens=False,
    #                 return_tensors="pt",
    #                 **tokenizer_kwargs
    #             )
    #         else:
    #             prefix = self.args.prefix if self.args.prefix else self.args.padding_text
    #             encoded_prompt = self.tokenizer.encode(
    #                 prefix + prompt_text, add_special_tokens=False, return_tensors="pt"
    #             )
    #
    #         encoded_prompt = encoded_prompt.to(self.args.device)
    #
    #         if encoded_prompt.size()[-1] == 0:
    #             input_ids = None
    #         else:
    #             input_ids = encoded_prompt
    #
    #         # generate token by token
    #         # output_sequences = model.generate(
    #         #     input_ids=input_ids,
    #         #     max_length=args.length + len(encoded_prompt[0]),
    #         #     temperature=args.temperature,
    #         #     top_k=args.k,
    #         #     top_p=args.p,
    #         #     repetition_penalty=args.repetition_penalty,
    #         #     do_sample=True,
    #         #     num_return_sequences=args.search_time * 2  # args.num_return_sequences
    #         # )
    #         text_gen_dict = dict()
    #         text_gen_dict_no_boost = dict()
    #         id = 1
    #
    #         for i in range(1, num_sen * 20):
    #             text_gen, done = self.gen_token_by_token(
    #                 input_ids,
    #                 temperature=1,
    #                 features=features,
    #                 topk=topk,
    #                 syn_lst=syn_lst
    #             )
    #             # if done:
    #             print('--->', text_gen)
    #             if i <= num_sen:
    #                 text_gen_dict_no_boost[i] = '.'.join(text_gen.split('.')[:-1]) + '.'
    #             if done:
    #                 text_gen_dict[id] = text_gen
    #                     #'.'.join(text_gen.split('.')[story_sen_id:])
    #                 id += 1
    #                 if id > num_sen:
    #                     break
    #
    #         # append text w/o boosting into return dict
    #         id_start = len(text_gen_dict_res) + 1
    #         if not strict:
    #             id = 1
    #             for key in text_gen_dict:
    #                 text_gen_dict_res[id] = text_gen_dict[key]
    #                 id += 1
    #             for key in text_gen_dict_no_boost:
    #                 text_gen_dict_res[id] = text_gen_dict_no_boost[key]
    #                 id += 1
    #             count = 100
    #         else:
    #             for key in text_gen_dict:
    #                 text_gen_dict_res[id_start] = text_gen_dict[key]
    #                 id_start += 1
    #             if id_start > num_sen:
    #                 count = 100
    #
    #     return text_gen_dict_res
    #
    # def choose_from_top(self, probs, n):
    #     ind = np.argpartition(probs, -n)[-n:]
    #     top_prob = probs[ind]
    #     top_prob = top_prob / np.sum(top_prob)  # Normalize
    #     choice = np.random.choice(n, 1, p=top_prob)
    #     token_id = ind[choice][0]
    #     return int(token_id)
    #
    # def gen_token_by_token(self, cur_token_ids, temperature, features, syn_lst, topk=5):
    #     previous_id = 0
    #     done_sig = 0
    #     len_prompt = cur_token_ids.size()[-1]
    #
    #     for i in range(100 + len_prompt):
    #         # print('cur_token_ids', cur_token_ids)
    #         outputs = self.model(input_ids=cur_token_ids)
    #         # print('outputs',outputs)
    #         logits = outputs[0]
    #         next_word_logits = logits[0, -1] / temperature
    #         softmax_logits = torch.softmax(next_word_logits, dim=0)
    #
    #         probs = softmax_logits.to("cpu").detach().numpy()
    #
    #         # boost the prob based on comet features
    #         probs, done = self.boost_obj.boost(self.tokenizer, probs, features, previous_id=previous_id)
    #         next_token_id = self.choose_from_top(probs, n=topk)
    #
    #         previous_id = next_token_id
    #
    #
    #         # append to cur tokens
    #         next_token_id_int = next_token_id
    #         next_token_id = torch.tensor([[next_token_id]]).to(self.args.device)
    #         cur_token_ids = torch.cat((cur_token_ids, next_token_id), dim=-1)
    #
    #         if done:  # only return finishing boost when we can find the feature in the text
    #             res = self.check_done(syn_lst,
    #                                text=self.tokenizer.decode(list(cur_token_ids[0:1, len_prompt:].to("cpu").detach().numpy())[0]))
    #             if res:
    #                 done_sig = 1  # boost activated
    #
    #
    #         if (done_sig and [next_token_id_int] in self.eos) or i >= 40:
    #             break
    #     # print(cur_token_ids[0:1, len_prompt:])
    #     output_text = self.tokenizer.decode(list(cur_token_ids[0:1, len_prompt:].to("cpu").detach().numpy())[0])
    #     # print(output_text)
    #     if done_sig:
    #         # print('res', res, done_sig)
    #         return res, done_sig
    #     else:
    #         # print('output_text', output_text, done_sig)
    #         return output_text, done_sig

    def features_dealt(self, list_of_list, chars_names, prompt_text=None):
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

    def generate_continuation(self, file_path, file_write):
        file_write = open(file_write, 'w')
        with open(file_path, 'r') as f:
            for i, row in enumerate(f):
                if i >= 0:
                    sentences = row.split('.')
                    for s in sentences:
                        if s.strip() and 'Char' in s:

                            # check what char in this sentence
                            if '[Char_1]' in s and '[Char_2]' not in s:
                                char_s = 0
                            elif '[Char_1]' in s and '[Char_2]' in s:
                                char_s = 2
                            elif '[Char_1]' not in s and '[Char_2]' in s:
                                char_s = 1
                            else:
                                pass

                            for prefix_char in ['s', 'm']:
                                sig_lst = [True, True]  # # wrong-1 ; true-0
                                while True:
                                    text = self.gen_text(prompt_text=s, order_remain=1, prefix_char=prefix_char, char_s=char_s)

                                    if sig_lst[0] and self.check_match(s.strip(), text, self.comet, beam_size=10, char_type=prefix_char):  # wrong 1
                                        write_text = '1 *' +  s.strip() + '.' + '*' + text + '\n'
                                        file_write.write(write_text)
                                        print(write_text)
                                        file_write.write(self.replace_all(write_text))
                                        file_write.flush()
                                        sig_lst[0] = False

                                    elif sig_lst[1] and not self.check_match(s.strip(), text, self.comet, beam_size=10, char_type=prefix_char):  # true 0
                                        write_text = '0 *' + s.strip() + '.' + '*' + text + '\n'
                                        file_write.write(write_text)
                                        print(write_text)
                                        file_write.write(self.replace_all(write_text))
                                        file_write.flush()
                                        sig_lst[1] = False
                                    print(sig_lst)
                                    if not sig_lst[0] and not sig_lst[1]:
                                        break
        file_write.close()

    def finetune_model(self, round):
        self.model = run_finetune_gpt2(gpu_use='0', pho=0.001, round=round, device=self.args.device)


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
        default='finetuned_model/roc_char/21',
        type=str,
        required=False,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--length", type=int, default=20)
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

    parser.add_argument("--seed", type=int, default=1, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument("--device_id",
                        type=int,
                        default=0,
                        help="gpu id")
    parser.add_argument(
        "--comet_checkpoints",
        type=str,
        default="../../comet-atomic-2020/models/comet_atomic2020_bart/comet-atomic_2020_BART",
    )
    args = parser.parse_args()
    args.device = torch.device(
        "cuda:" + str(args.device_id) if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else 1

    set_seed(args)

    gpt_2_gen = GPT_2_gen(args)
    print('args.device', args.device)
    # output = gpt_2_gen.gen_multiple_obj(prompt_text='Jenny lived in Florida.", "Jenny hears',
    #                                     num=3)
    # output = gpt_2_gen.calculate_prob(sentences=['Jenny studied hard. Jenny and her husband during testing did well.', 'Jenny studied hard. Jenny is a rock star.'])
    # output = gpt_2_gen.gen_text_decoding(prompt_text='I lived in Florida.', num_sen=3, features=['beach'], story_sen_id=1, topk=10, strict=False)
    # print('output => ', output)
    # print(gpt_2_gen.gen_text(prompt_text='* [Char_1] * [Char_1] loves ice cream.', order_remain=1, output_obj=False, restriction=None))
    # gpt_2_gen.generate_continuation('../data_story/100KStories_dealed.txt', 'data/continuation.txt')

    i = 0
    # gpt_2_gen.generate_continuation('../data_story/test_story.txt', 'data/continuation_' + str(i) + '.txt')
    gpt_2_gen.finetune_model(round=i)
    # print(gpt_2_gen.replace_all(text='[Char_1] made cake for [Char_2]'))