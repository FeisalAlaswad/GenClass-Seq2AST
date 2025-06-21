import os
from pathlib import Path

import pandas as pd
import json
import math
import re
from collections import OrderedDict

import json
import os

from src.ast_class_descriptor import ASTClassDescriptor
from src.plantuml_ast_processor import PlantUMLASTProcessor


def detect_format(file_path):
    if file_path.endswith(".jsonl"):
        return "jsonl"
    elif file_path.endswith(".json"):
        return "json"
    else:
        raise ValueError(f"Unsupported file format for: {file_path}")


def load_json_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json_file(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_jsonl_file(lines, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in lines:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def merge_multiple_files(file_paths, output_path):
    if not file_paths:
        raise ValueError("No files provided for merging.")

    # Detect format of the first file
    first_format = detect_format(file_paths[0])

    # Check all files have the same format
    for fp in file_paths[1:]:
        if detect_format(fp) != first_format:
            raise ValueError("All files must be in the same format (json or jsonl).")

    if first_format == "json":
        merged = None
        for fp in file_paths:
            print(fp)
            data = load_json_file(fp)
            if merged is None:
                merged = data
            else:
                if isinstance(merged, list) and isinstance(data, list):
                    merged += data
                elif isinstance(merged, dict) and isinstance(data, dict):
                    merged = {**merged, **data}
                else:
                    raise ValueError("All JSON files must be of the same type (list or dict).")
        write_json_file(merged, output_path)

    elif first_format == "jsonl":
        merged_lines = []
        for fp in file_paths:
            lines = load_jsonl_file(fp)
            merged_lines.extend(lines)
        write_jsonl_file(merged_lines, output_path)



def extract_title(input_str):
    # Match from number until either the '-' or the end of the string
    match = re.match(r'^\d+\s(.*?)(?=\s-\s|$)', input_str)
    if match:
        return match.group(1)
    return ''


def is_valid(value):
    # Check if the value is a valid non-NaN value
    if isinstance(value, str):
        # Check if it's not an empty string or the string "NaN"
        return bool(value.strip()) and value != "NaN"
    elif isinstance(value, (int, float)):
        # Check if it's not NaN for numeric values
        return not math.isnan(value)
    return False


def process_json_and_add_ast(input_file_path: str, output_json_path: str, output_jsonl_path: str):
    """
    Loads a JSON file, parses each 'PlantUML' field into an AST using PlantUMLASTProcessor,
    adds the 'Output_AST' to each element, and saves the result as both a JSON and JSONL file.

    Args:
        input_file_path (str): Path to the input JSON file.
        output_json_path (str): Path to save the full JSON array.
        output_jsonl_path (str): Path to save the data as JSON Lines (JSONL).
    """
    # Load the input JSON file
    with open(input_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Process each element in the list
    for element in data:
        plantuml_code = element.get("PlantUML", "")
        try:
            output_ast = PlantUMLASTProcessor.parse_plantuml_to_ast(plantuml_code)
            element["Output_AST"] = output_ast
        except Exception as e:
            print(f"Error processing element at index {element.get('RequirementIndex')}: {e}")
            element["Output_AST"] = None

    # Save full JSON
    with open(output_json_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    # Save as JSONL
    with open(output_jsonl_path, "w", encoding="utf-8") as file:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            file.write(json_line + "\n")


# Function to process each directory
def generate_json_dataset(directory, output_path):
    json_data = []
    json_data2 = []
    # List all directories in the given path
    directories = [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]

    # Sort directories based on the numeric part of the folder name (with error handling)
    directories.sort(
        key=lambda folder: int(folder.split(" ", 1)[0]) if folder.split(" ", 1)[0].isdigit() else float('inf'))

    # Iterate over the sorted directories
    for folder in directories:
        folder_path = os.path.join(directory, folder)

        # Extract the number from the folder name
        folder_parts = folder.split(" ", 1)
        if len(folder_parts) == 2 and folder_parts[0].isdigit():
            number = folder_parts[0]

            # Initialize the paths for the files
            xlsx_file = os.path.join(folder_path, f"{number}.xlsx")
            txt_file = os.path.join(folder_path, f"{number}.txt")
            png_file = os.path.join(folder_path, f"{number}.png")

            # Check if the necessary files exist
            if os.path.exists(xlsx_file) and os.path.exists(txt_file) and os.path.exists(png_file):
                # Read the xlsx file into a dataframe
                df = pd.read_excel(xlsx_file)

                # Extract HumanLang and PlantUML columns
                human_lang = df['HumanLang'].tolist()
                plant_uml = df['PlantUML'].tolist()
                index = 0
                # Build the JSON structure
                for hl, pu in zip(human_lang, plant_uml):
                    if is_valid(hl) and is_valid(pu):
                        json_data.append({
                            "HumanLang": hl,
                            "PlantUML": pu,
                            "Model": f"C{number}",
                            "RequirementIndex": f"{index}",
                            "Output_AST": PlantUMLASTProcessor.parse_plantuml_to_ast(pu)
                        })
                    index = index + 1

        json_data2.append({
            "Model": f"C{number}",
            "Image": f"{number}.png",
            "FullCode": f"{number}.txt",
            "Title": extract_title(folder).strip(),

        })

    # Save to JSON file
    with open(output_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

    with open(str(output_path) + "l", "w") as jsonl_file:
        for record in json_data:
            jsonl_file.write(json.dumps(record) + "\n")  # Write each record as a line

    with open(str(os.path.dirname(output_path)) + "/raw_index.jsonl", "w") as jsonl_file2:
        for record in json_data2:
            jsonl_file2.write(json.dumps(record) + "\n")  # Write each record as a line

    print(f"JSON file saved to {output_path}")



def json_to_txt_specific_field(json_file_path, field, output_path, distinct=False):
    with open(json_file_path, "r", encoding="utf-8") as file:
        json_data = json.load(file)

    seen = set()
    with open(output_path, "w", encoding="utf-8") as file:
        for item in json_data:
            if field in item:
                line = item[field].replace("\n", "\\n")
                if distinct:
                    if line in seen:
                        continue
                    seen.add(line)
                file.write(line + "\n")

    print(f"Text file saved to {output_path}")



def replace_json_field_with_txt_lines(json_file_path, txt_file_path, field, output_json_path, output_jsonl_path):
    # Read JSON data (assume a list of dicts)
    with open(json_file_path, 'r', encoding='utf-8') as f_json:
        data = json.load(f_json)

    # Read all lines from the text file (strip to remove trailing newlines)
    with open(txt_file_path, 'r', encoding='utf-8') as f_txt:
        txt_lines = [line.rstrip('\n') for line in f_txt]

    # Replace the specified field's value with corresponding txt line by index
    for i, item in enumerate(data):
        if i < len(txt_lines):
            if field in item:
                item[field] = txt_lines[i].strip()
            else:
                # Optionally handle missing field case, here we skip
                pass
        else:
            # No corresponding line in txt file; optionally leave as is or clear field
            # For now, we leave it unchanged
            pass

    # Save updated data back to JSON file
    with open(output_json_path, 'w', encoding='utf-8') as f_out_json:
        json.dump(data, f_out_json, ensure_ascii=False, indent=2)

    # Save updated data to JSONL file (one JSON object per line)
    with open(output_jsonl_path, 'w', encoding='utf-8') as f_out_jsonl:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            f_out_jsonl.write(json_line + '\n')

# Example usage:
# replace_json_field_with_txt_lines('input.json', 'input.txt', 'fieldName', 'output.json', 'output.jsonl')





# Provide the root directory where the folders are located
#root_directory = Path(__file__).parent.parent / "data/raw/dataset/PlantUCD train - formatted"
#root_directory = Path(__file__).parent.parent / r"data\raw\dataset test\dataset\PlantUCD test"
#output_path = Path(__file__).parent.parent / "data/raw/test.json"
#generate_json_dataset(root_directory,output_path)


"""
json_to_txt_specific_field(r"F:\GenClass\data\raw\raw test\description1.json",
                           "HumanLang",
                           r"F:\GenClass\data\raw\raw test\HumanLang_tagged.txt",
                           distinct=False)



replace_json_field_with_txt_lines(r"F:\GenClass\data\raw\raw test\description1 old.json",
                                  r"F:\GenClass\data\raw\raw test\HumanLang_tagged.txt",
                                  "HumanLang",
                                  r"F:\GenClass\data\raw\raw test\description1.json",
                                  r"F:\GenClass\data\raw\raw test\description1.jsonl")



# Create description.json and description.jsonl from raw.json as description of class ast
with open(r"F:\GenClass\data\raw\dataset test\test.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert each AST to human language
for entry in data:
    ast = entry.get("Output_AST")
    if ast:
        entry["HumanLang"] = ASTClassDescriptor.ast_to_nl(ast)

# Save to JSON file
with open(r"F:\GenClass\data\raw\dataset test\description2.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

# Save to JSONL file
with open(r"F:\GenClass\data\raw\dataset test\description2.jsonl", "w", encoding="utf-8") as f:
    for entry in data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


"""

merge_multiple_files([
    r"F:\GenClass\data\raw\raw train\aug1.json",
    r"F:\GenClass\data\raw\raw train\aug2.json",
    r"F:\GenClass\data\raw\raw train\aug3.json",
    r"F:\GenClass\data\raw\raw train\description1.json",
    r"F:\GenClass\data\raw\raw train\description2.json",
    r"F:\GenClass\data\raw\raw train\train.json",

     ],
    r"F:\GenClass\data\raw\raw train\PlantUCD_dataset_train.json", )

merge_multiple_files([
    r"F:\GenClass\data\raw\raw train\aug1.jsonl",
    r"F:\GenClass\data\raw\raw train\aug2.jsonl",
    r"F:\GenClass\data\raw\raw train\aug3.jsonl",
    r"F:\GenClass\data\raw\raw train\description1.jsonl",
    r"F:\GenClass\data\raw\raw train\description2.jsonl",
    r"F:\GenClass\data\raw\raw train\train.jsonl",
     ],
    r"F:\GenClass\data\raw\raw train\PlantUCD_dataset_train.jsonl", )


"""
process_json_and_add_ast( r"F:\GenClass\data\raw\aug old.json",
                          r"F:\GenClass\data\raw\aug1.json",
                          r"F:\GenClass\data\raw\aug1.jsonl")

"""
