import pickle, random
f_survey = open("surveys/beam_2numMatch.txt", "w+")
f_survey_idx = open("surveys/beam_idx_2numMatch.txt", "w+")

f_survey.write("[[AdvancedFormat]]\n")

question_begin = "[[Question:Matrix]] \nPlease read the following stories. Once you are done, you will determine which story is better based on character actions and logic.\n\n"
choices = "[[Choices]]\n In which story are the actions of the main character more consistent and logical?\n"
pre_answers = "[[AdvancedAnswers]]\n"
# story_list = pickle.load(open("story_list.p", "rb"))
# random.shuffle(story_list)
# pickle.dump(story_list[:20], open("gen_story.p", "wb+"))

# gen_story = pickle.load(open("gen_story.p", "rb"))

# checker_1 = ("christmas was very soon. [FEMALE] was nervous about christmas. she had not eaten in a long time. she felt very sick that night. [FEMALE] decided to be more healthy. ",
#              "christmas was very soon. [FEMALE] and her friends were going to have a party. they had invited a bunch of friends. they were going to have a party together. it was going to be a great party. ",
#              "christmas was very soon. [FEMALE] and her friends were going to have a party. they had invited a bunch of friends. they were going to have a party together. it was going to be a great party. ")
#
# gen_story.insert(4, checker_1)
#
# checker_2 = ("[MALE] was cooking bacon. but he accidentally dropped it. he did n’t know what to do. in his anger , he grabbed a knife. and he stabbed the knife in his mouth. ",
#              "[MALE] was cooking bacon. he was making a big pot of bacon. he added the bacon to the pot. [MALE] started to cook the bacon quickly. [MALE] got the bacon out of the pot.",
#              "[MALE] was cooking bacon. he was making a big pot of bacon. he added the bacon to the pot. [MALE] started to cook the bacon quickly. [MALE] got the bacon out of the pot.")
# gen_story.insert(15, checker_2)
gen_story_base_name = "./data_story/story_gen_fl_strong_ex_chars_history_noname.txt"
gen_story_want_name = "./data_story/story_gen_fl_strong_ex_chars_history_noname_numMat2.txt"
gen_story_base = open(gen_story_base_name , "r")
gen_story_want = open(gen_story_want_name, "r")

f_survey_idx.write(gen_story_base_name + "\n")
f_survey_idx.write(gen_story_want_name + "\n")

index = 1
story_pair = []
for baseline, want in zip(gen_story_base, gen_story_want):
    # if index == 12:
    #     f_survey.write("[[Block]]\n\n")
    f_survey.write(question_begin)
    index += 1
    f_survey.write(choices)
    f_survey.write(pre_answers)
    li_story = [baseline, want]
    random.shuffle(li_story)
    idx = 0
    for story in li_story:
        if story == baseline:
            f_survey_idx.write(str(idx))
        idx += 1

        f_survey.write("[[Answer]]\n")
        formatted_story = "<div style=\"text-align: left;\">" + \
                          story.replace(". ", ".<br/> ").replace("\n", "") + "</div>\n"
        f_survey.write(formatted_story)
    f_survey.write("\n\n")
