import re

def read_and_format_yaml(filename):
    formatted_data = []
    with open(filename, 'r') as file:
        content = file.read()
        sections = [s.strip() for s in content.split('\n\n') if s.strip()]

        for section in sections:
            input_match = re.search('input: (.+)', section)
            output_match = re.search('output: (.+)', section)
            cot_match = re.search('cot: (.+)', section)
            pred_output_match = re.search('pred_output: (.+)', section)

            if not (input_match and output_match and cot_match and pred_output_match):
                continue

            input_str = input_match.group(1)
            output_str = output_match.group(1)
            cot_str = cot_match.group(1)
            pred_output_str = pred_output_match.group(1)

            # 转换字符串为列表
            input_list = eval(input_str)
            output_list = eval(output_str)
            cot_list = eval(cot_str)

            # 转换pred_output字符串为cot格式
            pred_output_lists = pred_output_str.split(' <###> ')[:-1]  # 去除末尾的空字符串
            pred_output_lists = [[word.strip("['\"") for word in po_list.split(' ') if word] for po_list in pred_output_lists]

            formatted_data.append({
                'input': input_list,
                'output': output_list,
                'cot': cot_list,
                'pred_output': pred_output_lists
            })

    return formatted_data

def dump_to_yaml(data, filename):
    with open(filename, 'w') as file:
        for item in data:
            for key, value in item.items():
                file.write(f"{key}: {value}\n")
            file.write("\n")

inputname = "Visualization.yaml"
oututname = "Visualization_trans.yaml"

formatted_data = read_and_format_yaml(inputname)
dump_to_yaml(formatted_data, oututname)