from config import raw_train_data_jsonl_path, train_data_path,ecore_data_path
from config import test_data_path, raw_ecore_asts_jsonl_path,raw_test_data_jsonl_path
from config import raw_genmymodel_asts_jsonl_path, genmymodel_data_path,meged_train_data_path
from src.data_processor import DataProcessor  # update path to match your project
from src.data_processor_for_suggestions import DataProcessorForSuggestions
import json
import warnings
def build_multitask_data():

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        processor = DataProcessor(
        raw_file=raw_train_data_jsonl_path,
        top_nouns=12,
        top_verbs=12,
        )

        processor.run(output_file=train_data_path)

        processor = DataProcessor(
            raw_file=raw_test_data_jsonl_path,
            top_nouns=12, top_verbs=12,
        )
        processor.run(output_file=test_data_path)

        processor = DataProcessorForSuggestions(
            raw_file=raw_ecore_asts_jsonl_path,
        )

        processor.run(output_file=ecore_data_path)

        processor = DataProcessorForSuggestions(
            raw_file=raw_genmymodel_asts_jsonl_path,
        )
        processor.run(output_file=genmymodel_data_path)





    file_paths = [ecore_data_path, genmymodel_data_path, train_data_path]

    # Initialize an empty list to collect all data
    merged_data = []

    # Load and append contents from each file
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                merged_data.extend(data)  # Merge lists
            else:
                merged_data.append(data)  # Wrap dict as one item

    # Save the merged data to a new file
    with open(meged_train_data_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)


