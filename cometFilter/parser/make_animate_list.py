import pickle
l_animate = []

family_f = open("animate/family.txt", "r")
l_animate.extend([x.replace("\n", "") for x in family_f.readlines()])
relation_f = open("animate/relationships.txt", "r")
l_animate.extend([x.replace("\n", "") for x in relation_f.readlines()])

pickle.dump(l_animate, open('animate.p', 'wb+'))