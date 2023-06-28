import yaml
import re

def parse_yaml(file_path):
    data_list = []
    with open(file_path, "r") as file:
        yaml_data = file.read().split("\n\n")
        for yaml_string in yaml_data:
            data = dict(re.findall(r"(\w+): (.+)", yaml_string))
            for key in data.keys():
                data[key] = eval(data[key])
            data_list.append(data)
    return data_list

def read_yaml(filename):
    with open(filename, 'r') as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data

def find_differences(yaml_data):
    diff_data = []
    for data in yaml_data:
        if data['pred_output'] != data['cot']:
            diff_data.append({
                'input': data['input'],
                'output': data['output'],
                'cot': data['cot'],
                'pred_output': data['pred_output'],
                'matching': False
            })
        else:
            data['matching'] = True
            diff_data.append(data)
    return diff_data

def dump_to_yaml(data, filename):
    with open(filename, 'w') as file:
        for item in data:
            for key, value in item.items():
                if key == 'matching' and not value:
                    file.write(f"{key}: {value}\n")
                else:
                    file.write(f"{key}: {value}\n" if not isinstance(value, list)
                                else f"{key}: {', '.join(map(str, value))}\n")
            file.write("\n")



if __name__ == "__main__":
    file_name = 'Visualization_trans.yaml' # 更改为你的输入文件名
    yaml_data = parse_yaml(file_name)
    diff_data = find_differences(yaml_data)
    output_file_name = 'Visualization_trans_compare.yaml'  # 更改为你要存储的输出文件名
    dump_to_yaml(diff_data, output_file_name)
