from collections import Counter
import re
import json
from collections import Counter
import re


#[('entities', 292), ('verbs', 292), ('manage', 133), ('allow', 119), ('support', 111), ('track', 88), ('access', 83), ('create', 76), ('enable', 65), ('maintain', 62), ('register', 61), ('monitor', 56), ('log', 55), ('management', 53), ('user', 51), ('application', 51), ('information', 50), ('software', 50), ('generate', 49), ('provide', 48), ('view', 47), ('payment', 45), ('functionality', 45), ('include', 43), ('process', 41), ('notify', 40), ('record', 38), ('inventory', 37), ('store', 36), ('handle', 36), ('email', 35), ('customer', 33), ('tracking', 31), ('receive', 31), ('admin', 31), ('update', 30), ('schedule', 30), ('upload', 29), ('account', 28), ('facilitate', 28), ('platform', 28), ('modify', 27), ('add', 27), ('submit', 26), ('control', 26), ('ensure', 25), ('send', 24), ('authentication', 24), ('security', 23), ('need', 23)]
#['entities', 'verbs', 'manage', 'allow', 'support', 'track', 'access', 'create', 'enable', 'maintain', 'register', 'monitor', 'log', 'management', 'user', 'application', 'information', 'software', 'generate', 'provide', 'view', 'payment', 'functionality', 'include', 'process', 'notify', 'record', 'inventory', 'store', 'handle', 'email', 'customer', 'tracking', 'receive', 'admin', 'update', 'schedule', 'upload', 'account', 'facilitate', 'platform', 'modify', 'add', 'submit', 'control', 'ensure', 'send', 'authentication', 'security', 'need']
from src.ast_class_descriptor import ASTClassDescriptor


def get_most_frequent_words_from_file(filepath, top_n=10):
    # Read paragraph from file
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    # Normalize and tokenize the text
    words = re.findall(r'\b\w+\b', text.lower())

    # Count word frequencies
    word_counts = Counter(words)

    # Get top N most common words
    return word_counts.most_common(top_n)




filepath = 'F:/GenClass/data/raw/scenario words.txt'
top_words = get_most_frequent_words_from_file(filepath, top_n=50)
words_only = [word for word, _ in top_words]
print(top_words)
print(words_only)

