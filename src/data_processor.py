import sys
import random

from src.name_formatter import NameFormatter

main_dir='/content/drive/MyDrive/GenClass'
sys.path.append(main_dir)
src_dir='/content/drive/MyDrive/GenClass/src'
sys.path.append(src_dir)

import json
from collections import defaultdict, Counter
import re
import string
from nltk.stem import PorterStemmer

from src.keybert_extractor import KeybertExtractor
from src.plantuml_ast_processor import PlantUMLASTProcessor


class DataProcessor:
    predefined_stopwords = [
        "a", "an", "the", "and", "or", "but", "if", "then", "else",
        "when", "while", "for", "to", "from", "of", "in", "on", "at", "by",
        "with", "without", "about", "into", "over", "under", "after", "before",
        "be", "is", "are", "was", "were", "been", "being", "have", "has", "had",
        "having", "do", "does", "did", "doing",
        "will", "would", "shall", "should", "can", "could", "may", "might", "must",
        "that", "which", "who", "whom", "whose", "this", "these", "those",
        "each", "every", "some", "any", "all", "both", "few", "many", "more", "most", "other",
        "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
        "as", "just", "also", "yet", "even", "still",
    ]

    def __init__(self, raw_file, stopwords=None, top_nouns=10,top_verbs=10, history_length=10):
        self.raw_file = raw_file
        self.stopwords = stopwords or self.predefined_stopwords
        self.top_nouns = top_nouns
        self.top_verbs = top_verbs
        self.model_to_items = defaultdict(list)
        self.model_context = {}
        self.stemmer = PorterStemmer()
        self.history_length = history_length
        self.avoid_words = ['manage', 'allow', 'support', 'track', 'access', 'create', 'enable',
                       'maintain', 'register',
                       'monitor', 'log', 'management', 'user', 'application', 'information', 'software', 'generate',
                       'provide', 'view',
                       'payment', 'functionality', 'include', 'process', 'notify', 'record', 'inventory', 'store',
                       'handle', 'email',
                       'customer', 'tracking', 'receive', 'admin', 'update', 'schedule', 'upload', 'account',
                       'facilitate', 'platform',
                       'modify', 'add', 'submit', 'control', 'ensure', 'send', 'authentication', 'security', 'need',
                            'methods','method','attribute','attributes','class']
    def load_data(self):
        with open(self.raw_file, 'r', encoding='utf-8') as f:
            for line in f:
                if len(line.strip())>0:
                    item = json.loads(line)
                    self.model_to_items[item['Model']].append(item)

    def tokenize(self, text):
        # Simple tokenizer: split by space and remove punctuation
        tokens = re.findall(r'\b\w+\b', text.lower())

        # Stem, remove stopwords, remove punctuation, and filter out 1-character words
        filtered_tokens = set()  # Use a set to avoid duplicates

        for t in tokens:
            if t not in self.stopwords and t not in string.punctuation and len(t) > 1:
                stemmed_word = self.stemmer.stem(t)
                filtered_tokens.add(stemmed_word)

        return list(filtered_tokens)

    def build_model_context_by_keybert(self):
        for model, items in self.model_to_items.items():
            all_text = " ".join(item['HumanLang'] for item in items)

            filtered_words = [
                word for word in re.findall(r'\b\w+\b', all_text)
                if word.lower() not in set(w.lower() for w in self.avoid_words)
            ]

            # Reconstruct the string
            filtered_text = ' '.join(filtered_words)
            all_text=filtered_text
            # Extract top nouns and verbs
            top_keywords_nouns = KeybertExtractor.extract_keywords_nouns(all_text, top_nouns=self.top_nouns)
            top_keywords_verbs = KeybertExtractor.extract_keywords_verbs(all_text, top_verbs=self.top_verbs)

            # Pad or trim nouns
            if len(top_keywords_nouns) < self.top_nouns:
                top_keywords_nouns += ['<PAD>'] * (self.top_nouns - len(top_keywords_nouns))
            else:
                top_keywords_nouns = top_keywords_nouns[:self.top_nouns]

            # Pad or trim verbs
            if len(top_keywords_verbs) < self.top_verbs:
                top_keywords_verbs += ['<PAD>'] * (self.top_verbs - len(top_keywords_verbs))
            else:
                top_keywords_verbs = top_keywords_verbs[:self.top_verbs]

            # Format final context string
            formatted_context = (
                    "Entities: " + ', '.join(top_keywords_nouns) + "\n" +
                    "Verbs: " + ', '.join(top_keywords_verbs)
            )

            # Store in model context dictionary
            self.model_context[model] = formatted_context


    @staticmethod
    def extract_all_class_names(ast_node):
        classes = []
        if isinstance(ast_node, list):
            for node in ast_node:
                classes.extend(DataProcessor.extract_all_class_names(node))
        elif isinstance(ast_node, dict):
            if ast_node.get("type") == "class" and ast_node.get("value") and "children" in ast_node:
                classes.append(ast_node["value"])
            for val in ast_node.values():
                if isinstance(val, (list, dict)):
                    classes.extend(DataProcessor.extract_all_class_names(val))
        return classes


    @staticmethod
    def extract_class_names_have_children_only(ast_json):
        """
        Extracts a list of class names from the given AST JSON.

        Parameters:
            ast_json (dict): The JSON object following the AST UML Schema.

        Returns:
            List[str]: A list of class names.
        """
        if not isinstance(ast_json, dict):
            raise ValueError("Input must be a JSON object (Python dict).")

        if ast_json.get("type") != "root":
            raise ValueError("Root node must have type 'root'.")

        class_names = []

        for child in ast_json.get("children", []):
            if isinstance(child, dict) and child.get("type") == "class":
                class_names.append(child.get("value"))

        return class_names

    @staticmethod
    def extract_attributes_for_class(ast_json, class_name):
        """
        Extracts formatted attributes for a specific class from the AST JSON.

        Parameters:
            ast_json (dict): The AST JSON object.
            class_name (str): Name of the class to extract attributes from.

        Returns:
            List[str]: A list of formatted attribute strings.
        """
        visibility_map = {
            "+": "[PUBLIC]",
            "-": "[PRIVATE]",
            "#": "[PROTECTED]",
            "~": "[PACKAGE]"
        }

        if ast_json.get("type") != "root":
            raise ValueError("Invalid AST root type.")

        for class_node in ast_json.get("children", []):
            if class_node.get("type") == "class" and class_node.get("value") == class_name:
                attributes = []
                for child in class_node.get("children", []):
                    if child.get("type") == "attribute":
                        visibility_symbol = child.get("visibility", "+").strip()  # default to '+'
                        visibility_tag = visibility_map.get(visibility_symbol, "[PUBLIC]")

                        name = child.get("value", "unnamedAttr")
                        data_type = child.get("data_type", "string")
                        name = NameFormatter.to_camel_case_attr(name)
                        #attr_str = f"[ATTRIBUTE] {visibility_tag} {name} [TYPE] {data_type} [/ATTRIBUTE]"
                        attr_str = f"{visibility_tag} {name} [TYPE] {data_type}"
                        attributes.append(attr_str)
                return attributes

        return []  # Class not found

    @staticmethod
    def extract_methods_for_class(ast_json, class_name):
        """
        Extracts formatted methods for a specific class from the AST JSON.

        Parameters:
            ast_json (dict): The AST JSON object.
            class_name (str): Name of the class to extract methods from.

        Returns:
            List[str]: A list of formatted method strings.
        """
        visibility_map = {
            "+": "[PUBLIC]",
            "-": "[PRIVATE]",
            "#": "[PROTECTED]",
            "~": "[PACKAGE]"
        }



        if ast_json.get("type") != "root":
            raise ValueError("Invalid AST root type.")

        for class_node in ast_json.get("children", []):
            if class_node.get("type") == "class" and class_node.get("value") == class_name:
                methods = []
                for child in class_node.get("children", []):
                    if child.get("type") == "method":
                        visibility_symbol = child.get("visibility", "+").strip()  # default to '+'
                        visibility_tag = visibility_map.get(visibility_symbol, "[PUBLIC]")

                        name = child.get("value", "unnamedMethod")
                        name = NameFormatter.to_camel_case_method(name)
                        data_type = child.get("data_type", "void")


                        #method_str = f"[METHOD] {visibility_tag} {name} [TYPE] {data_type} [/METHOD]"
                        method_str = f"{visibility_tag} {name} [TYPE] {data_type}"
                        methods.append(method_str)
                return methods

        return []  # Class not found



    @staticmethod
    def extract_relations_for_class(ast_json, class_name):
        """
        Extracts formatted relations from all classes in the AST JSON.

        Parameters:
            ast_json (dict): The AST JSON object.

        Returns:
            List[str]: A list of formatted relation strings.
        """

        def inverse_relation(symbol: str) -> str:
            # Reverse the string
            reversed_symbol = symbol[::-1]

            # Swap '<' and '>' characters
            swapped = []
            for ch in reversed_symbol:
                if ch == '>':
                    swapped.append('<')
                elif ch == '<':
                    swapped.append('>')
                else:
                    swapped.append(ch)

            return ''.join(swapped)

        if ast_json.get("type") != "root":
            raise ValueError("Invalid AST root type.")

        # Mapping for relation types
        rel_map = {
            "--": "[ASSOCIATION]",

            "<|--": "[RIGHT:EXTENSION]",
            "--|>": "[LEFT:EXTENSION]",

            "<|..": "[RIGHT:IMPLEMENTATION]",
            "..|>": "[LEFT:IMPLEMENTATION]",

            "*--": "[LEFT:COMPOSITION]",
            "--*": "[RIGHT:COMPOSITION]",

            "o--": "[LEFT:AGGREGATION]",
            "--o": "[RIGHT:AGGREGATION]",

            "-->": "[LEFT:ASSOCIATION]",
            "<--": "[RIGHT:ASSOCIATION]",

            "..": "[DEPENDENCY]",
            "..>": "[LEFT:DEPENDENCY]",
            "<..": "[RIGHT:DEPENDENCY]"
        }

        # Mapping for multiplicities
        def format_multiplicity(mult):
            return {
                "1": "[ONE]",
                "*": "[MANY]",
                "0..*": "[MOM]",
                "1..*": "[OOM]",
                "0..1": "[ZOO]",
            }.get(mult, "[ONE]")  # Default to [ONE] if unrecognized

        relations = []



        for class_node in ast_json.get("children", []):
            if class_node.get("type") == "class" and class_node.get("value") == class_name:
                class_name = class_node.get("value", "Class1")
                for child in class_node.get("children", []):
                    if child.get("type") == "relation":
                        relation_type = child.get("value", "--")
                        label = child.get("label", "[NOLABEL]")
                        multiplicity1 = format_multiplicity(child.get("multiplicity1", "1"))
                        multiplicity2 = format_multiplicity(child.get("multiplicity2", "1"))
                        related_class_node = child.get("children", [{}])[0]
                        related_class_name = related_class_node.get("value", "Class2")


                        label = label.replace("_"," ").strip('"')
                        label = NameFormatter.to_camel_case_attr(label)
                        source_class = NameFormatter.to_pascal_case(class_name)
                        target_class = NameFormatter.to_pascal_case(related_class_name)

                        if random.choice([True, False]) and (len(label) == 0 or label == "[NOLABEL]"):
                            tag = rel_map.get(inverse_relation(relation_type), "[ASSOCIATION]")
                            relation_str = (
                                f"{tag} {target_class} {multiplicity1} {multiplicity2} {NameFormatter.to_pascal_case(source_class)} "
                                f"[LABEL] {label}"
                            )
                        else:
                            tag = rel_map.get(relation_type, "[ASSOCIATION]")
                            relation_str = (
                                f"{tag} {source_class} {multiplicity1} {multiplicity2} {NameFormatter.to_pascal_case(target_class)} "
                                f"[LABEL] {label}"
                            )

                        relations.append(relation_str)

        return relations

    def preprocess(self):
        preprocessed = []
        for model, items in self.model_to_items.items():
            scenario_context = self.model_context[model]
            sorted_items = sorted(items, key=lambda x: int(x['RequirementIndex']))

            seen_class_names = []
            history_length = getattr(self, "history_length", 20)
            classes_of_model=[]
            relations_of_model = []
            relations_by_classes = {}
            for item in sorted_items:
                input_sentence = item['HumanLang'].strip()
                output_ast = item['Output_AST']


                # history_context is class names from previous items
                history_context = seen_class_names[:history_length]

                # Extract class names from current AST recursively
                new_classes = list(set(DataProcessor.extract_all_class_names(output_ast)) - set(seen_class_names))
                classes_of_model.extend(new_classes)
                # Update seen_class_names keeping order and uniqueness
                for cls in new_classes:
                    seen_class_names.append(cls)


                classes = DataProcessor.extract_class_names_have_children_only(output_ast)
                preprocessed.append({
                    'input_sentence': input_sentence,
                    'scenario_context': scenario_context,
                    'history_context': history_context,
                    'task': "Class Identification",
                    #'output_ast_tagged': "[AST:SEG]\n"+'\n'.join(f'[CLASS] {NameFormatter.uppercase_first_letter(name)} [/CLASS]' for name in classes)+"\n[/AST:SEG]]",
                    'output_ast_tagged': "[AST:SEG]\n" + '\n'.join(
                        f'{NameFormatter.uppercase_first_letter(name)}' for name in
                        classes) + "\n[/AST:SEG]]"

                })
                preprocessed.append({
                    'input_sentence': input_sentence,
                    'scenario_context': scenario_context,
                    'history_context': ["[NOHISTORY]"],
                    'task': "Class Identification",
                    #'output_ast_tagged': "[AST:SEG]\n"+'\n'.join(f'[CLASS] {NameFormatter.uppercase_first_letter(name)} [/CLASS]' for name in classes)+"\n[/AST:SEG]]"
                    'output_ast_tagged': "[AST:SEG]\n" + '\n'.join(
                        f'{NameFormatter.uppercase_first_letter(name)}' for name in
                        classes) + "\n[/AST:SEG]]"
                })

                for cls in classes:
                    attributes_of_class = DataProcessor.extract_attributes_for_class(output_ast,cls)
                    preprocessed.append({
                        'input_sentence': input_sentence,
                        'scenario_context': scenario_context,
                        'history_context': history_context,
                        'task': "Attribute Identification",
                        "class": cls,
                        'output_ast_tagged': "[AST:SEG]\n"+'\n'.join(f'{attr}' for attr in attributes_of_class)+"\n[/AST:SEG]]"
                    })

                    methods_of_class  = DataProcessor.extract_methods_for_class(output_ast, cls)
                    preprocessed.append({
                        'input_sentence': input_sentence,
                        'scenario_context': scenario_context,
                        'history_context': history_context,
                        'task': "Method Identification",
                        "class": cls,
                        'output_ast_tagged': "[AST:SEG]\n"+'\n'.join(f'{method}' for method in methods_of_class)+"\n[/AST:SEG]]"
                    })

                    relations_of_class = DataProcessor.extract_relations_for_class(output_ast, cls)

                    preprocessed.append({
                        'input_sentence': input_sentence,
                        'scenario_context': scenario_context,
                        'history_context': history_context,
                        'task': "Relation Identification",
                        "class": cls,
                        'output_ast_tagged':  "[AST:SEG]\n" + "\n".join(relations_of_class) + "\n[/AST:SEG]]"
                    })

                    if cls not in relations_by_classes:
                        relations_by_classes[cls] = []  # Initialize list for this class
                    relations_by_classes[cls].extend(relations_of_class)
                    relations_of_model.extend(relations_of_class)



            classes_of_model = list(dict.fromkeys(classes_of_model))
            relations_of_model = list(dict.fromkeys(relations_of_model))

            """
            for cls, relations_list in relations_by_classes.items():
                relations_list = list(dict.fromkeys(relations_list))
                preprocessed.append({
                    'scenario_context': scenario_context,
                    'class': cls,
                    'task': "Class Relation Identification",
                    'output_ast_tagged': "[AST:SEG]\n" + "\n".join(relations_list) + "\n[/AST:SEG]]"
                })
            """

        return preprocessed

    def save_processed(self, processed_data, output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)

    def run(self, output_file):
        self.load_data()
        self.build_model_context_by_keybert()
        processed_data = self.preprocess()
        self.save_processed(processed_data, output_file)
        print(f"Saved {len(processed_data)} processed samples to {output_file}")
