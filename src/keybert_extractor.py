from keybert import KeyBERT
import spacy

# Load spaCy for POS tagging and lemmatization
nlp = spacy.load("en_core_web_sm")
kw_model = KeyBERT()

class KeybertExtractor:

    @staticmethod
    def extract_keywords_nouns(text: str, top_nouns: int = 10):
        doc = nlp(text)
        seen_noun_lemmas = set()
        noun_lemmas_ordered = []

        for token in doc:
            if token.pos_ == "NOUN":
                lemma = token.lemma_
                if lemma not in seen_noun_lemmas:
                    seen_noun_lemmas.add(lemma)
                    noun_lemmas_ordered.append(lemma)

        scored_nouns = kw_model.extract_keywords(
            text,
            candidates=noun_lemmas_ordered,
            stop_words='english',
            use_mmr=False,
            diversity=0.6,
            top_n=top_nouns
        )

        noun_set = {kw for kw, _ in scored_nouns}
        return [kw for kw in noun_lemmas_ordered if kw in noun_set]

    @staticmethod
    def extract_keywords_verbs(text: str, top_verbs: int = 10):
        doc = nlp(text)
        seen_verb_lemmas = set()
        verb_lemmas_ordered = []

        for token in doc:
            if token.pos_ == "VERB":
                lemma = token.lemma_
                if lemma not in seen_verb_lemmas:
                    seen_verb_lemmas.add(lemma)
                    verb_lemmas_ordered.append(lemma)

        scored_verbs = kw_model.extract_keywords(
            text,
            candidates=verb_lemmas_ordered,
            stop_words='english',
            use_mmr=False,
            diversity=0.6,
            top_n=top_verbs
        )

        verb_set = {kw for kw, _ in scored_verbs}
        return [kw for kw in verb_lemmas_ordered if kw in verb_set]


