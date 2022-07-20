f_write = open('../data_story/prompt_deal.txt', 'w')
with open('../data_story/100KStories_dealed.txt', 'r') as f:
    for story in f:
        sentences = story.split('.')[:-1]
        for idx, sentence in enumerate(sentences):
            if idx < len(sentences) - 1:
                char = []
                if 'Char_1' in sentences[idx + 1]:
                    char.append('Char_1')
                if 'Char_2' in sentences[idx + 1]:
                    char.append('Char_2')
                for c in char:
                    res = '*' + c + '*' + sentence + '#' + sentences[idx + 1]
                    f_write.write(res)
f_write.close()