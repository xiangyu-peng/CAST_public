import sys
sys.path.append("../comet-atomic-2020/models/comet_atomic2020_bart")
from generation_example import Comet
import argparse
from nltk_gen import nltk_gen
from similarity import *
from collections import defaultdict
from tqdm import tqdm, trange

criteria_matching = dict()
# criteria_matching['s'] = [['xWant', 'xIntent'], ['xEffect', 'sentence'], ['CausesDesire', 'Desires'],
#                 ['isBefore', 'isAfter'], ['AtLocation', 'AtLocation'], ['sentence', 'xNeed'], ['xReact', 'xAttr'], ['none', 'none']]
criteria_matching['s'] = [['xWant', 'xIntent'], ['xReact', 'xReact'], ['CausesDesire', 'Desires'],
                ['xEffect', 'xEffect'], ['xReact', 'xAttr'], ['none', 'none']]
# criteria_matching['m'] = [['oReact', 'xAttr'], ['oWant', 'xIntent'], ['oEffect', 'sentence'], ['none', 'none']]
criteria_matching['m'] = [['oReact', 'xAttr'], ['oWant', 'xIntent'], ['oEffect', 'xEffect'], ['none', 'none']]

def filter_obj_comet(list_of_objects, comet_model, nltk_model, prompt):
    set_of_entities = set()
    for obj in list_of_objects:
        queries = ["{} {} [GEN]".format(obj, rel) for rel in comet_model.all_relations]
        results = comet_model.generate(queries, decode_method="beam", num_generate=5)
        for result in results:
            for entity in result:
                entity = nltk_model.comet_to_delete(words=entity, sentence=prompt)
                if entity and entity != 'none':
                    set_of_entities.add(entity.replace('.', '').strip().lower())
        return set_of_entities

def check_matching_criteria(results_lst, sentences):
    '''
    Given inference pairs and find best criteria pair
    :param results_lst: [inf_prompt, inf_cont]; inf_prompt:dict()---key -> rel_comet; value -> inference
    :param sentences: [prompt, continuation]
    :return: criteria_pair_lst
    '''
    criteria_pair = defaultdict(float)
    prom_result, cont_result = results_lst
    prom_result['sentence'] = [sentences[0]]
    cont_result['sentence'] = [sentences[1]]

    for key_1 in prom_result:
        for key_2 in cont_result:
            prompt_inf = prom_result[key_1]
            cont_inf = cont_result[key_2]
            prompt_inf_new = []
            content_inf_new = []
            for i, content in enumerate(prompt_inf):
                if content.strip() and 'none' not in content.lower():
                    prompt_inf_new.append(content.lower().replace('personx', '').replace('persony', ''))

            for i, content in enumerate(cont_inf):
                if content.strip() and 'none' not in content.lower():
                    content_inf_new.append(content.lower().replace('personx', '').replace('persony', ''))

            score = similarity_score_max(prompt_inf_new, content_inf_new)
            criteria_pair[(key_1, key_2)] = score
            # print(key_1, key_2, score)
            # print(prompt_inf_new)
            # print(content_inf_new)

    criteria_pair_lst = [(key, criteria_pair[key]) for key in criteria_pair]
    criteria_pair_lst = sorted(criteria_pair_lst, key=lambda x:x[-1], reverse=True)
    return criteria_pair_lst

def read_file_matching_criteria(file_path, chars='s'):
    '''
    Read files and check the matching criteria
    :param file_path: path of txt file. each line has one story.
    :return: summary_dict
    '''
    summary_dict = defaultdict(float)
    count = 0
    story_count = 0
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for line in tqdm(lines):
        if (chars == 's' and '[Char_2]' not in line):
            story_count += 1
            print('->', story_count)
            sentences = line.split('.')
            for idx in range(len(sentences) - 2):
                # Generate inferences from comet 2020.
                results_lst = get_comet_result(comet, sentences[idx:idx+2])
                # Given inference pairs and find best criteria pair
                criteria_pair_lst = check_matching_criteria(results_lst=results_lst,
                                                            sentences=sentences[idx:idx+2])
                # Update summary
                for tup in criteria_pair_lst:
                    summary_dict[tup[0]] = (summary_dict[tup[0]] * count + tup[-1]) / (count + 1)
                count += 1
        if chars == 'm' and '[Char_2]' in line:
            sentences = line.split('.')
            print('->', story_count)
            story_used = False
            for idx in range(len(sentences) - 2):
                if ('[Char_1]' in sentences[idx] and '[Char_2]' in sentences[idx+1]) or \
                    ('[Char_2]' in sentences[idx] and '[Char_1]' in sentences[idx+1]):
                    # Generate inferences from comet 2020.
                    results_lst = get_comet_result(comet, sentences[idx:idx+2])
                    # Given inference pairs and find best criteria pair
                    criteria_pair_lst = check_matching_criteria(results_lst=results_lst,
                                                                sentences=sentences[idx:idx+2])
                    # Update summary
                    # print(criteria_pair_lst)
                    for tup in criteria_pair_lst:
                        summary_dict[tup[0]] = (summary_dict[tup[0]] * count + tup[-1]) / (count + 1)
                    count += 1
                    story_used = True
            if story_used:
                story_count += 1

        if story_count % 5 == 0:
            print('~')
        if story_count >= 100:
            break
    # print results
    summary_dict_lst = [(key, summary_dict[key]) for key in summary_dict]
    summary_dict_lst = sorted(summary_dict_lst, key=lambda x: x[-1], reverse=True)
    f_write = open(args.save_file_path, 'w')
    for key in summary_dict_lst:
        f_write.write(str(key[0]) + '===>' + str(key[1]) + '\n')
    f_write.close()
    return summary_dict

def get_comet_result(comet, sentences):
    '''
    Generate inferences from comet 2020.
    :param sentences: [prompt, continuation]
    :return: results_lst; [dict, dict]
    '''
    results_lst = []
    for head in sentences:
        queries = ["{} {} [GEN]".format(head, rel) for rel in comet.all_relations]
        results = comet.generate(queries, decode_method="beam", num_generate=5)
        results_dict = dict()
        for i, rel in enumerate(comet.all_relations):
            results_dict[rel] = results[i]
        results_lst.append(results_dict)
    return results_lst

def get_comet_result_single(comet, sentence, num_generate=10):
    '''
    Generate inferences from comet 2020.
    :param sentences: [prompt, continuation]
    :return: results_lst; [dict, dict]
    '''
    queries = ["{} {} [GEN]".format(sentence, rel) for rel in comet.all_relations]
    results = comet.generate(queries, decode_method="beam", num_generate=num_generate)
    results_dict = dict()
    for i, rel in enumerate(comet.all_relations):
        results_dict[rel] = [r for r in results[i] if 'none' not in r]
    return results_dict

def verify_criteria(results_lst, sentences, type='s'):
    '''
    verify the matching criteria. Pls see https://github.gatech.edu/xpeng62/CommonsenseStoryGen/tree/inlg#matching-criteria
    :param type: str; 's' or 'm': 's' - single; and 'm' - 'multiple'
    :return:
    '''
    if type == 's':
        criteria = [['xWant', 'xIntent'], ['sentence', 'xNeed'], ['xEffect', 'sentence'], ['CausesDesire', 'Desires'], ['isBefore', 'isAfter'], ['AtLocation', 'AtLocation']]
    elif type =='m':
        criteria = [['oReact', 'xAttr'], ['oWant', 'xIntent'], ['oEffect', 'sentence']]
    else:
        raise ValueError("type has to be 's' or 'm'.")

    prom_result, cont_result = results_lst
    for key_1, key_2 in criteria:
        if key_1 == 'sentence':
            prompt_inf = sentences[0]
        else:
            prompt_inf = prom_result[key_1]

        if key_2 == 'sentence':
            cont_inf = sentences[1]
        else:
            cont_inf = cont_result[key_2]

        print('>>>', sentences[0])
        print(key_1)
        print(prompt_inf)
        print('-' * 10)
        print('>>>', sentences[1])
        print(key_2)
        print(cont_inf)
        print('-' * 5)
        print('Similarity =>', similarity_score_max(prompt_inf, cont_inf))
        print('=*' * 30)

def comet_2020_Filter(comet, sentence, type='prompt', num_generate=10, char_type='s', decode=False):
    '''
    Get sentence inference for cometFilter.
    :param sentence: str.
    :param type: prompt or continuation
    :param comet: comet_model
    :param type: prompt or continuations
    :param num_generate: # of each rel generated
    :param char_type: s or m; single or multiple
    :param decode: whether to use for decoding, if yes, we do not consider any sentences it self.
    :return: list of list
    '''
    res = []
    results_dict = get_comet_result_single(comet, sentence, num_generate=num_generate)
    idx = 0 if type == 'prompt' else 1
    end_idx = -1 if not decode else 3
    for rel in criteria_matching[char_type][:end_idx]:
        rel = rel[idx]
        if rel == 'sentence':
            res.append([sentence])
        else:
            res.append(results_dict[rel])
    return res

def load_model(checkpoints):
    return Comet(checkpoints)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", type=str, default="", help="prompt and continuation")
    parser.add_argument("--char", type=int, default=1, help="number of char")
    parser.add_argument("--verify", action='store_true', help="whether you want to verify the criteria here")
    parser.add_argument("--check", action='store_true', help="whether you want to check the matching criteria here")
    parser.add_argument("--file_check", action='store_true', help="whether you want to check the matching criteria in the file")
    parser.add_argument("--story_file_path",
                        type=str,
                        default='data_story/100KStories_dealed.txt',
                        help="file path of story you use for matching criteria.")
    parser.add_argument("--save_file_path",
                        type=str,
                        default='data_story/criteria_result.txt',
                        help="file path of story you save results for matching criteria.")

    args = parser.parse_args()
    nltk_gen = nltk_gen()

    if args.example:
        prompt = args.example.split('.')[0] + '.'
        continuation = args.example.split('.')[1] + '.'
    else:  # you can also define ur prompt and continuations here
        prompt = 'Mike proposed to Linda.'
        continuation = 'Linda said yes.'
    sentences = [prompt, continuation]

    # path of trained weight of COMeT (no need to edit, download in https://github.com/allenai/comet-atomic-2020)
    comet = load_model("../comet-atomic-2020/models/comet_atomic2020_bart/comet-atomic_2020_BART")  # new atomic comet 2020
    comet.model.zero_grad()

    if args.verify:
        if args.char == 1:
            args.char = 's'
        else:
            args.char = 'm'

        verify_criteria(results_lst, sentences, type=args.char)

    if args.check:
        check_matching_criteria(results_lst, sentences)

    if args.file_check:
        read_file_matching_criteria(args.story_file_path)

    a = comet_2020_Filter(comet=comet,
                      sentence='[Char_1] had a big crush on [Char_2].',
                      type='prompt',
                      num_generate=10,
                      char_type='m')

    b = comet_2020_Filter(comet=comet,
                            sentence='[Char_2] wanted to go to prom with [Char_1].',
                            type='continuation',
                            num_generate=10,
                            char_type='m')

    # print(filter_obj_comet(list_of_objects=['sleep', 'sleepy'], comet_model=comet, nltk_model=nltk_gen, prompt='John is sleepy.'))
    for i in range(len(a)):
        print(a[i])
        print(b[i])
        print('-' * 30)