scores = input("Score line for a model ")
scores = scores.split('|')

for i in range(len(scores) - 1, -1, -1):
    if ')' not in scores[i]:
        del scores[i]


sumScores = 0.0
for score in scores:
    score = score.split(',')
    print(score[4])



#print(scores)


