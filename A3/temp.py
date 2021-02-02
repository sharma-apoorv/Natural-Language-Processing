import numpy as np

def relu(x):
    return np.maximum(x, 0)

def score_q2(theta, word_emb_list):
    score = 0
    word_emb_sum = relu(sum(word_emb_list))
    # print(sum(word_emb_list), word_emb_sum, theta)
    score = np.dot(theta, word_emb_sum)

    return score

# Assume V = [good, bad, not]
# _good = np.array([1,  0.3368, -0.1973])
# _bad = np.array([0.2726, 1, 0.3233])
# _not = np.array([0, 0, 1])

# theta = np.array([12, -2, -4])

_good = np.array([3, 1, 2, -4])
_bad = np.array([0.5, 0.5, 0.5, -1])
_not = np.array([-3, -1, -2, 4])

theta = np.array([3, 4, 4, 20])

print(f"good: {score_q2(theta, [_good])}")
# print(f"not: {score_q2(theta, [_not])}")
print(f"not good: {score_q2(theta, [_not, _good])}")

print(f"bad: {score_q2(theta, [_bad])}")
print(f"not bad: {score_q2(theta, [_not, _bad])}")
