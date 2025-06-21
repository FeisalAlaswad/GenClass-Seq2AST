import sys

from src.data_processor import DataProcessor
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


class DataProcessorForSuggestions:

    def __init__(self, raw_file):
        self.raw_file = raw_file
        self.items = []  # store individual items

    def load_data(self):
        with open(self.raw_file, 'r', encoding='utf-8') as f:
            for line in f:
                if len(line.strip()) > 0:
                    item = json.loads(line)
                    self.items.append(item)


    def preprocess(self):
        preprocessed = []
        for item in self.items:

            output_ast = item['Output_AST']
            classes = DataProcessor.extract_all_class_names(output_ast)
            sorted_classes =sorted(set(classes))
            scenario_context = f"Entities: {', '.join(sorted_classes)}"
            for cls in sorted_classes:
                attributes_of_class = DataProcessor.extract_attributes_for_class(output_ast, cls)
                if len(attributes_of_class) != 0:
                    preprocessed.append({
                    'scenario_context': scenario_context,
                    'task': "Attribute Suggestion",
                    'class': cls,
                    'output_ast_tagged': "[AST:SEG]\n" + '\n'.join(f'{attr}' for attr in attributes_of_class) + "\n[/AST:SEG]]"
                    })

                methods_of_class = DataProcessor.extract_methods_for_class(output_ast, cls)
                if len(methods_of_class) != 0:
                    preprocessed.append({
                        'scenario_context': scenario_context,
                        'task': "Method Suggestion",
                        'class': cls,
                        'output_ast_tagged': "[AST:SEG]\n" + '\n'.join(
                            f'{method}' for method in methods_of_class) + "\n[/AST:SEG]]"
                    })


                relations_of_class = DataProcessor.extract_relations_for_class(output_ast, cls)
                if len(relations_of_class) != 0:
                    preprocessed.append({
                        'scenario_context': scenario_context,
                        'task': "Relation Suggestion",
                        'class': cls,
                        'output_ast_tagged': "[AST:SEG]\n" + "\n".join(relations_of_class) + "\n[/AST:SEG]]"
                    })
        return preprocessed

    def save_processed(self, processed_data, output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)

    def run(self, output_file):
        self.load_data()
        processed_data = self.preprocess()
        self.save_processed(processed_data, output_file)
        print(f"Saved {len(processed_data)} processed samples to {output_file}")


