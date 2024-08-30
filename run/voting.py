import json
import yaml
import argparse
from collections import Counter
from tqdm import tqdm

def load_paths_from_yaml(yaml_file):
    with open(yaml_file, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config['paths']

def hard_voting(paths, output_file):
    reference_file = 'resource/data/대화맥락추론_test.json'

    with open(reference_file, 'r', encoding='utf-8') as ref_file:
        reference_data = json.load(ref_file)

    data_list = []
    for path in paths:
        with open(path, 'r', encoding='utf-8') as file:
            data_list.append(json.load(file))

    if any(len(reference_data) != len(data) for data in data_list):
        raise ValueError("The files do not contain the same number of entries.")

    total_entries = len(reference_data)

    hard_voting_results = []

    for i in range(total_entries):
        outputs = [data[i]['output'] for data in data_list]
        most_common_output = Counter(outputs).most_common(1)[0][0]
        result_entry = reference_data[i].copy()
        result_entry['output'] = most_common_output
        hard_voting_results.append(result_entry)

    with open(output_file, 'w', encoding='utf-8') as output_file_handle:
        json.dump(hard_voting_results, output_file_handle, ensure_ascii=False, indent=4)

    print(f"Hard voting results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Hard Voting Script")
    parser.add_argument("--yaml_file", type=str, required=True, help="Path to the YAML file containing the paths")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the hard voting results")

    args = parser.parse_args()

    paths = load_paths_from_yaml(args.yaml_file)
    
    hard_voting(paths, args.output_file)

if __name__ == "__main__":
    main()
