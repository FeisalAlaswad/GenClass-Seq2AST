# config.py

RAW_DATA_DIR = '/content/drive/MyDrive/GenClass/data/raw'
PROCESSED_DATA_DIR = '/content/drive/MyDrive/GenClass/data/processed'
SCHEMA_DIR = '/content/drive/MyDrive/GenClass/schemas'
CHECK_POINTS_DIR = '/content/drive/MyDrive/GenClass/check-points'


raw_train_data_path = f"{RAW_DATA_DIR}/PlantUCD_dataset_train.json"
raw_train_data_jsonl_path = f"{RAW_DATA_DIR}/PlantUCD_dataset_train.jsonl"

raw_test_data_path = f"{RAW_DATA_DIR}/PlantUCD_dataset_test.json"
raw_test_data_jsonl_path = f"{RAW_DATA_DIR}/PlantUCD_dataset_test.jsonl"

raw_genmymodel_asts_path = f"{RAW_DATA_DIR}/genmymodel_asts.json"
raw_genmymodel_asts_jsonl_path = f"{RAW_DATA_DIR}/genmymodel_asts.jsonl"

raw_ecore_asts_path = f"{RAW_DATA_DIR}/ecore_asts.json"
raw_ecore_asts_jsonl_path = f"{RAW_DATA_DIR}/ecore_asts.jsonl"

train_data_path = f"{PROCESSED_DATA_DIR}/train.json"
test_data_path = f"{PROCESSED_DATA_DIR}/test.json"
genmymodel_data_path = f"{PROCESSED_DATA_DIR}/genmymodel.json"
ecore_data_path = f"{PROCESSED_DATA_DIR}/ecore.json"
meged_train_data_path = f"{PROCESSED_DATA_DIR}/merged_train_data.json"

schema_path = f"{SCHEMA_DIR}/json_schema.json"

grammars_file_path = f"{SCHEMA_DIR}/grammers.txt"
saved_check_point_path = f"{CHECK_POINTS_DIR}/codet5-genclass3"

epochs = 20
batch_size=16
model_name = "Salesforce/codet5-small"