from lark import Lark
from lark.exceptions import UnexpectedInput
from ast_generator import ASTGenerator
from config import grammars_file_path, test_data_path, batch_size, saved_check_point_path, model_name

if __name__ == "__main__":
    # Path to your grammar file
    file_path = grammars_file_path

    # Open and read the file
    with open(file_path, 'r', encoding='utf-8') as file:
        grammar = file.read()

    parser = Lark(grammar, start='start')
    # Initialize generator
    generator = ASTGenerator(model_name=model_name,
                             parser=parser,
                             load_dir_path=saved_check_point_path)

    generator.evaluate_bleu_on_data(test_data_path,
        batch_size=batch_size,
        num_samples=100)
