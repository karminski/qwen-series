#!/usr/bin/env python3
"""
将Qwen模型CSV文件转换为JSON格式时间线
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import re

def convert_csv_to_json():
    """将CSV文件转换为JSON格式"""

    csv_file = Path("qwen_models_cleaned.csv")
    if not csv_file.exists():
        print(f"错误: 找不到文件 {csv_file}")
        return None

    print("开始读取CSV文件...")

    # 读取CSV文件
    df = pd.read_csv(csv_file)
    print(f"成功读取 {len(df)} 行数据")

    models = []

    for _, row in df.iterrows():
        name = str(row['name']).strip()
        size = str(row['size']).strip()
        model_type = str(row['type']).strip()
        update_time = str(row['update_time']).strip()
        downloads = str(row['downloads']).strip()

        # 跳过空行
        if not name or name == 'nan':
            continue

        # 解析下载量数字
        download_num = parse_download_number(downloads)

        # 标准化时间
        time_standardized = standardize_time(update_time)

        # 解析模型大小
        size_parsed = parse_model_size(size)

        model_data = {
            "name": name,
            "size": size_parsed,
            "type": model_type,
            "update_time": update_time,
            "update_time_standardized": time_standardized,
            "downloads": downloads,
            "downloads_number": download_num,
            "huggingface_url": f"https://huggingface.co/{name}"
        }

        models.append(model_data)

    print(f"成功处理 {len(models)} 个模型")

    # 按时间排序
    models_sorted = sorted(models, key=lambda x: x["update_time_standardized"], reverse=True)

    # 生成统计信息
    stats = generate_statistics(models)

    # 构建最终JSON结构
    json_data = {
        "metadata": {
            "total_models": len(models),
            "generated_date": datetime.now().isoformat(),
            "source_file": str(csv_file),
            "time_range": stats["time_range"],
            "types": stats["types"],
            "sizes": stats["sizes"]
        },
        "statistics": stats,
        "models": models_sorted
    }

    # 保存JSON文件
    json_file = Path("qwen_models_timeline.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"JSON数据已保存到: {json_file}")
    return json_data

def parse_download_number(download_str):
    """解析下载量字符串为数字"""
    if not download_str or download_str == '—' or download_str == 'nan':
        return 0

    download_str = str(download_str).strip()

    try:
        # 直接转换为整数
        return int(float(download_str))
    except:
        return 0

def standardize_time(time_str):
    """标准化时间格式用于排序"""
    if not time_str or time_str == '—' or time_str == 'nan':
        return "9999-99-99"  # 缺失时间排到最后

    time_str = str(time_str).strip()

    # 处理具体日期格式
    date_formats = [
        '%Y-%m-%d',       # 2024-01-25
        '%b %d, %Y',      # Jan 25, 2024
        '%B %d, %Y',      # January 25, 2024
        '%d %b %Y',       # 25 Jan 2024
        '%d %B %Y',       # 25 January 2024
    ]

    for fmt in date_formats:
        try:
            parsed_date = datetime.strptime(time_str, fmt)
            return parsed_date.strftime('%Y-%m-%d')
        except:
            continue

    # 如果无法解析，返回原值用于显示
    return time_str

def parse_model_size(size_str):
    """解析模型大小"""
    if not size_str or size_str == '—' or size_str == 'nan':
        return None

    size_str = str(size_str).strip()

    # 提取数字部分
    match = re.match(r'(\d+(?:\.\d+)?)', size_str)
    if match:
        return float(match.group(1))

    return None

def generate_statistics(models):
    """生成统计信息"""

    # 时间范围
    times = [m["update_time_standardized"] for m in models if m["update_time_standardized"] != "9999-99-99" and not m["update_time_standardized"].startswith("0")]
    time_range = {
        "start": min(times) if times else None,
        "end": max(times) if times else None
    }

    # 类型统计
    types = {}
    for model in models:
        model_type = model["type"]
        types[model_type] = types.get(model_type, 0) + 1

    # 大小统计
    sizes = {}
    for model in models:
        size = model["size"]
        if size:
            sizes[size] = sizes.get(size, 0) + 1

    # 下载量统计
    total_downloads = sum(m["downloads_number"] for m in models)
    avg_downloads = total_downloads / len(models) if models else 0

    return {
        "time_range": time_range,
        "types": types,
        "sizes": sizes,
        "total_downloads": total_downloads,
        "avg_downloads": avg_downloads,
        "models_with_time": len([m for m in models if m["update_time_standardized"] != "9999-99-99"]),
        "models_with_size": len([m for m in models if m["size"] is not None])
    }

if __name__ == "__main__":
    try:
        json_data = convert_csv_to_json()

        if json_data:
            print("\n=== 数据摘要 ===")
            print(f"总模型数: {json_data['metadata']['total_models']}")
            print(f"时间范围: {json_data['statistics']['time_range']['start']} - {json_data['statistics']['time_range']['end']}")
            print(f"模型类型数: {len(json_data['statistics']['types'])}")
            print(f"不同大小数: {len(json_data['statistics']['sizes'])}")
            print(f"有时间的模型: {json_data['statistics']['models_with_time']}")
            print(f"有大小的模型: {json_data['statistics']['models_with_size']}")

            print("\n模型类型分布:")
            for model_type, count in json_data['statistics']['types'].items():
                percentage = (count / json_data['metadata']['total_models']) * 100
                print(f"  {model_type}: {count} ({percentage:.1f}%)")

            print(f"\n总下载量: {json_data['statistics']['total_downloads']:,}")
            print(f"平均下载量: {json_data['statistics']['avg_downloads']:.1f}")

    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()