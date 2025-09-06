import os
import json

# 设置你的文件夹路径
folder_path = r'K:\Re-code\chachacha\deeplabv3-plus-pytorch-main\datasets\before'  # 例如："./labels"

# 遍历所有 .json 或 .txt 文件
for file_name in os.listdir(folder_path):
    if file_name.endswith(".json") or file_name.endswith(".txt"):
        file_path = os.path.join(folder_path, file_name)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            modified = False
            if "shapes" in data:
                for shape in data["shapes"]:
                    if shape.get("label") == "Air Conditioner Conderser":
                        shape["label"] = "Air Conditioner Condenser"
                        modified = True

            if modified:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                print(f"✅ 已修正标签：{file_name}")
            else:
                print(f"✅ 无需修改：{file_name}")

        except Exception as e:
            print(f"❌ 处理失败：{file_name}，错误：{e}")
