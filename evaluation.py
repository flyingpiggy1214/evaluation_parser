import spacy
import conllu
from conllu import parse_incr
from collections import defaultdict

# Load the English and Chinese treebanks
en_data_file = "UD_English-GUM/en_gum-ud-dev_test.conllu"
zh_data_file = "UD_Chinese-GSDSimp/zh_gsdsimp-ud-dev_test.conllu"

# Load the English and Chinese models
nlp_en = spacy.load("en_core_web_lg")
nlp_zh = spacy.load("zh_core_web_lg")

# Define evaluation functions
def evaluate(model, data):
    correct_labeled = 0
    correct_unlabeled = 0
    total = 0
    for sentence in parse_incr(data):
        text = " ".join([word["form"] for word in sentence])
        print(text)
        doc = model(text)
        for token, gold in zip(doc, sentence):
            print(token.text, token.dep_, gold["deprel"],token.head, token.head.i, gold["head"])
            if token.dep_ == gold["deprel"]:
                correct_unlabeled += 1
                if token.head.i + 1 == gold["head"]:
                    correct_labeled += 1
        total += len(sentence)
        print(correct_labeled)
        print(correct_unlabeled)
        print(total)
    uas = correct_unlabeled / total * 100
    las = correct_labeled / total * 100
    return uas, las

# Evaluate the English model
with open(en_data_file, "r", encoding="utf-8") as f:
    uas_en, las_en = evaluate(nlp_en, f)

print("English Model Evaluation Results")
print("UAS: {:.2f}%".format(uas_en))
print("LAS: {:.2f}%".format(las_en))

# Evaluate the Chinese model
with open(zh_data_file, "r", encoding="utf-8") as f:
    uas_zh, las_zh = evaluate(nlp_zh, f)

print("Chinese Model Evaluation Results")
print("UAS: {:.2f}%".format(uas_zh))
print("LAS: {:.2f}%".format(las_zh))
