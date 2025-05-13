# import json

# # 读取 forget.json
# with open('forget.json', 'r') as file:
#     data_forget = json.load(file)

# # 读取 retain1.json
# with open('retain1.json', 'r') as file:
#     data_retain = json.load(file)

# # 合并数据
# combined_data = data_forget + data_retain

# print(f"Total items: {len(combined_data)}")

# # 转换格式
# converted_data = [{"text": item} for item in combined_data]

# # 写入新的 JSON 文件（逐行存储）
# with open('full.json', 'w') as file:
#     for item in converted_data:
#         json.dump(item, file)
#         file.write("\n")  # 每个对象换行存储

# import json

# # 读取 all.txt 内容
# with open('all.txt', 'r', encoding='utf-8') as file:
#     lines = file.readlines()  # 逐行读取

# # 去除每行的换行符，并转换格式
# converted_data = [{"text": line.strip()} for line in lines if line.strip()]  # 过滤空行

# # 逐行写入 full.json，确保每个 JSON 对象独立存储
# with open('full.json', 'w', encoding='utf-8') as file:
#     for item in converted_data:
#         json.dump(item, file)
#         file.write("\n")  # 每个对象单独一行
import json
import random
# 计算要去除的前 3% 数据数量
# ratio_list = [0.01, 0.05, 0.10, 0.20]
# forget_name_list = ['01', '05', '10', '20']
# ratio_list = [0.30]
# forget_name_list = ['30']
ratio_list = [0.50, 0.90]
forget_name_list = ['50', '90']

# 读取 forget.txt
with open('forget.txt', 'r', encoding='utf-8') as file:
    forget_lines = set(line.strip() for line in file if line.strip())  # 去除空行并去重

# 读取 full.json
with open('full.json', 'r', encoding='utf-8') as file:
    full_data = [json.loads(line) for line in file]  # 按行解析 JSON 对象

# 提取 full.json 中的文本数据
full_texts = [entry["text"] for entry in full_data]

for ratio, forget_name in zip(ratio_list, forget_name_list):
    # 获取 full_data 对应的文本列表
    full_texts = [entry["text"] for entry in full_data]

    # 直接用索引找出要删除的项，避免过度删除
    to_remove_indices = [i for i, text in enumerate(full_texts) if text in forget_lines]

    if 3 * ratio > 1:
        remove_count = int(ratio * len(full_data))  # 确保 remove_count 是整数
        remaining_count = remove_count - len(to_remove_indices)

        # 如果仍然需要删除额外的数据
        if remaining_count > 0:
            # 找到 `full_texts` 中未在 `to_remove_indices` 里的索引，并随机采样
            remaining_indices = [i for i in range(len(full_texts)) if i not in to_remove_indices]
            sampled_indices = random.sample(remaining_indices, min(remaining_count, len(remaining_indices)))
            to_remove_indices.extend(sampled_indices)

    # 根据索引删除数据，保证删除数量准确
    filtered_data = [entry for i, entry in enumerate(full_data) if i not in to_remove_indices]

    # 生成要删除的 JSON 数据
    removed_data = [full_data[i] for i in to_remove_indices]

    # 打印去除前后的元素数
    print(f"Full.json 原始元素数: {len(full_data)}")
    print(f"Forget.txt 需要删除的元素数: {remove_count}")
    print(f"Forget.txt 匹配的元素数: {len(to_remove_indices)}")
    print(f"forget{forget_name}.json 处理后元素数: {len(filtered_data)}")

    # 存储去除后的 full.json 数据
    with open(f'full_minus_forget{forget_name}.json', 'w', encoding='utf-8') as file:
        for item in filtered_data:
            json.dump(item, file)
            file.write("\n")  # 逐行存储

    # 存储被去除的元素
    with open(f'forget{forget_name}.json', 'w', encoding='utf-8') as file:
        for item in removed_data:
            json.dump(item, file)
            file.write("\n")  # 逐行存储

