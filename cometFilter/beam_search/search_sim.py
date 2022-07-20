from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
import torch
import torch.nn.functional as F
import numpy as np
import pickle
import struct

def top_k_top_p_filtering(logits, top_k=20, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def cosine_similarity(vector1, vector2):
    vec_1_len = np.sqrt(np.sum(np.square(vector1)))
    vec_2_len = np.sqrt(np.sum(np.square(vector2)))
    return np.sum(np.multiply(vector1, vector2)) / (vec_1_len * vec_2_len)

def diverse_func(generated, logits, lamb, neural, exclude_words=[]):
    # Hamming distance
    if not neural:
        return hamming_distance(generated, logits, lamb, exclude_words=exclude_words)
    # return neural_hamming(generated, logits, lamb)
    return F.softmax(logits, dim=-1)

def hamming_distance(generated, logits, lamb, exclude_words=[]):
    generated = generated.flatten().tolist()
    for g in generated:
        w = list(vocab.items())[g][0]
        tfidf = 1
        if "Ġ" in w:
            w = w.replace("Ġ", "")
        if w not in exclude_words:
            if w.lower() in voc_idf:
                tfidf = voc_idf[w.lower()]
            if w.lower() in stopwords:
                tfidf = tfidf * lamb
            logits[:, g] = logits[:, g] - abs(logits[:, g] * lamb * tfidf)
    logits = F.softmax(logits, dim=-1)
    return logits

def neural_hamming(generated, logits, lamb):
    global sum_vectors
    # only penalize the top X choices
    # top 10 maybe?
    for _ in range(len(logits)):
        logits[_] = top_k_top_p_filtering(logits[_])
    # logits = top_k_top_p_filtering(logits)
    for k in range(len(logits)):
        for j in range(len(logits[0])):
            if logits[k][j] != float('-inf'):
                w = list(vocab.items())[j][0]
                if "Ġ" in w:
                    w = w.replace("Ġ", "")
                if w in word_model:
                    # iterate through everything generated
                    # and update accordingly
                    sim_sum = []
                    for gen_w, gen_vec in sum_vectors:
                        if (w, gen_w) in cosine_sim_dict:
                            sim_sum.append(cosine_sim_dict[(w, gen_w)])
                        elif (gen_w, w) in cosine_sim_dict:
                            sim_sum.append(cosine_sim_dict[(gen_w, w)])
                        else:
                            # not in there, gotta calculate
                            cos_sim_new = cosine_similarity(word_model[w], gen_vec)
                            cosine_sim_dict.update({(w, gen_w): cos_sim_new})
                            sim_sum.append(cos_sim_new)
                        if w == gen_w:
                            logits[k][j] = logits[k][j] - abs(logits[k][j] * lamb)
                    sim_factor = sum(sim_sum) / len(sim_sum)
                    logits[k][j] = logits[k][j] - abs(logits[k][j] * lamb * sim_factor)

    logits = F.softmax(logits, dim=-1)
    return logits

# The model should be passed in
def diverse_beam(model, length, raw_text, neural=False, num_samples=3, lamb=0.0005, temperature=1, exclude_words=[]):
    global sum_vectors, avg
    devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(devices)
    model.eval()

    # Setting up context
    context = tokenizer.encode(raw_text)
    # print('context', context)
    # Append to sum vectors, the list of vectors we have so far
    if neural:
        for c in context:
            w = list(vocab.items())[c][0]
            if "Ġ" in w:
                w = w.replace("Ġ", "")
            if w in word_model:
                sum_vectors.append((w, word_model[w]))

    context = torch.tensor(context, dtype=torch.long, device=devices)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    # Start sampling
    with torch.no_grad():
        for _ in range(length):
            inputs = {'input_ids': generated}
            # print('inputs')
            outputs = model(
                **inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            # print('outputs')
            # next_token_logits = F.softmax(outputs[0][:, -1, :] / temperature, dim=-1)
            next_token_logits = outputs[0][:, -1, :] / temperature
            # print('next_token_logits')

            # Add the diversity index
            # Add - lambda to words that are already in the set
            next_token_logits = diverse_func(generated, next_token_logits, lamb, neural, exclude_words=exclude_words)
            # print('next_token_logits')

            # filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(next_token_logits, num_samples=1).reshape(
                (num_samples, -1))
            # print('next_token')

            # next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=num_samples).reshape((num_samples, -1))
            if neural:
                for t in next_token.flatten().tolist():
                    w = list(vocab.items())[t][0]
                    if "Ġ" in w:
                        w = w.replace("Ġ", "")
                    if w in word_model:
                        sum_vectors.append((w, word_model[w]))
            # next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1).reshape(
                # (num_samples, -1))

            generated = torch.cat((generated, next_token), dim=1)
            # print('generated', generated)
    return generated


def decode_beam_search(generated, search_time):
    text_gen_dict = dict()
    sen_id = 0
    for gen in generated:
        gen = gen.tolist()
        text = tokenizer.decode(gen, clean_up_tokenization_spaces=True)
        text = text.replace("\n", ".")
        text = text.replace("?", ".")
        text = text.replace("!", ".")
        text = text.split(".")[1] + '.'
        tokens_len = len(tokenizer.encode(text))
        if tokens_len < 14 and tokens_len > 5:
            sen_id += 1
            text_gen_dict[sen_id] = text
            if sen_id >= search_time:
                return text_gen_dict

    return text_gen_dict
# Define beam size = num_samples

PATH_TO_BEAM_DATA = "/home/becky/Documents/CommonsenseStoryGen/cometFilter/beam_search/beam_data/"
MODEL_PATH = '/home/becky/Documents/CommonsenseStoryGen/cometFilter/finetune_language_model/finetuned_model/roc_char/done'
LAMBDA = 0.05

SEED_SENT = "* [Char_2] * [Char_1] noticed [Char_2] had put on a lot of weight recently."
EXCLUDE = ["David", "Luna"]

model_type = 'gpt2'
# Initialize Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_type)
vocab = tokenizer.get_vocab()
torch.manual_seed(0)

# Loading tf-idf
voc_idf = pickle.load(open(PATH_TO_BEAM_DATA + "voc_idf.p", "rb"))
# Neural word vector method
# Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)
# word_model = pickle.load(open(PATH_TO_BEAM_DATA + "glove.6B.300d.txt", "rb"))
def load_glove_model(File):
    print("Loading Glove Model")
    glove_model = {}
    with open(File,'r') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            glove_model[word] = embedding
    print(f"{len(glove_model)} words loaded!")
    return glove_model

# word_model = load_glove_model(PATH_TO_BEAM_DATA + "glove.6B.50d.txt")
word_model = load_glove_model('/home/becky/Documents/CommonsenseStoryGen/cometFilter/beam_search/6471382cdd837544bf3ac72497a38715e845897d265b2b424b4761832009c837/glove.6B.100d.txt')

sum_vectors = []
# avg = np.zeros((300,))
stopwords = pickle.load(open(PATH_TO_BEAM_DATA + "stopwords.p", "rb"))
cosine_sim_dict = dict()
# Need a diversity index
if __name__ == "__main__":
    model_class = GPT2LMHeadModel
    model = model_class.from_pretrained(MODEL_PATH)

    generated = diverse_beam(model, 18, SEED_SENT, num_samples=5, lamb=LAMBDA, exclude_words=EXCLUDE, neural=True)
    out = decode_beam_search(generated, 25)
    for gen in out.keys():
        print("[Char_1] noticed [Char_2] had put on a lot of weight recently.", out[gen])
    print("=================================")