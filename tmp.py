from gxl_ai_utils.utils import utils_file


lines = utils_file.load_list_file_clean("./data.list")
new_lines = []
for line in lines:
    line_new = line.replace(" <THINK> <LONG> <PUNCTUATION>", "")
    new_lines.append(line_new)
utils_file.write_list_to_file(new_lines, "./data.list")