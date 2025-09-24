import csv
import requests
import time
import re
from datetime import datetime
import concurrent.futures
from bs4 import BeautifulSoup
import threading
from typing import Optional, Dict, List
import random
import json

class HuggingFaceModelUpdater:
    def __init__(self, max_workers: int = 2, delay_range: tuple = (2, 4)):
        """
        初始化更新器

        Args:
            max_workers: 最大并发工作线程数
            delay_range: 请求延迟范围(秒)，用于防止被屏蔽
        """
        self.max_workers = max_workers
        self.delay_range = delay_range
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        self.progress_lock = threading.Lock()
        self.updated_count = 0
        self.failed_count = 0

    def extract_update_time_from_api(self, model_name: str) -> Optional[str]:
        """
        尝试使用Hugging Face API获取模型信息
        """
        api_url = f"https://huggingface.co/api/models/{model_name}"
        try:
            response = self.session.get(api_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # 检查是否有最后修改时间
                if 'lastModified' in data:
                    last_modified = data['lastModified']
                    # 解析ISO格式的时间
                    try:
                        dt = datetime.fromisoformat(last_modified.replace('Z', '+00:00'))
                        return dt.strftime("%Y-%m-%d")
                    except:
                        pass

                # 检查是否有创建时间或其他时间信息
                if 'createdAt' in data:
                    created_at = data['createdAt']
                    try:
                        dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        return dt.strftime("%Y-%m-%d")
                    except:
                        pass
            return None
        except:
            return None

    def extract_update_time_from_html(self, html_content: str) -> Optional[str]:
        """
        从HTML内容中提取更新时间
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            # 方法1: 查找包含"Updated"的文本
            updated_texts = soup.find_all(string=re.compile(r'Updated|Last updated|Modified', re.IGNORECASE))
            for text in updated_texts:
                text = text.strip()
                # 提取日期
                date_patterns = [
                    r'Updated\s+([A-Za-z]{3}\s+\d{1,2}(?:,\s+\d{4})?)',  # "Updated Aug 6" or "Updated Aug 6, 2025"
                    r'Updated\s+about\s+(.+?)\s+ago',                    # "Updated about 10 hours ago"
                    r'Last\s+updated\s+([A-Za-z]{3}\s+\d{1,2})',       # "Last updated Aug 6"
                    r'Modified\s+([A-Za-z]{3}\s+\d{1,2})',              # "Modified Aug 6"
                ]

                for pattern in date_patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        date_str = match.group(1)
                        # 解析日期
                        try:
                            if re.match(r'^[A-Za-z]{3}\s+\d{1,2}$', date_str):
                                # "Aug 6" 格式
                                current_year = datetime.now().year
                                full_date = f"{date_str} {current_year}"
                                parsed_date = datetime.strptime(full_date, "%b %d %Y")
                                return parsed_date.strftime("%Y-%m-%d")
                            elif re.match(r'^[A-Za-z]{3}\s+\d{1,2},\s+\d{4}$', date_str):
                                # "Aug 6, 2025" 格式
                                parsed_date = datetime.strptime(date_str, "%b %d, %Y")
                                return parsed_date.strftime("%Y-%m-%d")
                            elif 'ago' in date_str.lower():
                                # 相对时间，返回当前日期
                                return datetime.now().strftime("%Y-%m-%d")
                        except ValueError:
                            continue

            # 方法2: 查找meta标签中的时间信息
            meta_tags = soup.find_all('meta', attrs={'content': re.compile(r'\d{4}-\d{2}-\d{2}')})
            for tag in meta_tags:
                content = tag.get('content', '')
                date_match = re.search(r'(\d{4}-\d{2}-\d{2})', content)
                if date_match:
                    return date_match.group(1)

            # 方法3: 查找所有可能的日期格式
            date_patterns = [
                r'(\d{4}-\d{2}-\d{2})',                    # 2025-09-24
                r'([A-Za-z]{3}\s+\d{1,2},\s+\d{4})',       # Sep 24, 2025
                r'([A-Za-z]{3}\s+\d{1,2})',                 # Sep 24
                r'(\d{1,2}/\d{1,2}/\d{4})',               # 09/24/2025
            ]

            for pattern in date_patterns:
                matches = re.findall(pattern, html_content)
                if matches:
                    for match in matches:
                        try:
                            if re.match(r'^\d{4}-\d{2}-\d{2}$', match):
                                return match
                            elif re.match(r'^[A-Za-z]{3}\s+\d{1,2},\s+\d{4}$', match):
                                dt = datetime.strptime(match, "%b %d, %Y")
                                return dt.strftime("%Y-%m-%d")
                            elif re.match(r'^[A-Za-z]{3}\s+\d{1,2}$', match):
                                current_year = datetime.now().year
                                full_date = f"{match} {current_year}"
                                dt = datetime.strptime(full_date, "%b %d %Y")
                                return dt.strftime("%Y-%m-%d")
                            elif re.match(r'^\d{1,2}/\d{1,2}/\d{4}$', match):
                                dt = datetime.strptime(match, "%m/%d/%Y")
                                return dt.strftime("%Y-%m-%d")
                        except ValueError:
                            continue

            return None
        except Exception as e:
            print(f"解析HTML时出错: {e}")
            return None

    def fetch_model_update_time(self, model_name: str) -> Optional[str]:
        """
        获取单个模型的更新时间
        """
        # 首先尝试API
        api_time = self.extract_update_time_from_api(model_name)
        if api_time:
            with self.progress_lock:
                self.updated_count += 1
                print(f"✓ {model_name}: {api_time} (API)")
            return api_time

        # API失败，尝试网页爬取
        url = f"https://huggingface.co/{model_name}"

        try:
            # 添加随机延迟
            delay = random.uniform(*self.delay_range)
            time.sleep(delay)

            response = self.session.get(url, timeout=15)
            response.raise_for_status()

            update_time = self.extract_update_time_from_html(response.text)

            with self.progress_lock:
                if update_time:
                    self.updated_count += 1
                    print(f"✓ {model_name}: {update_time}")
                else:
                    self.failed_count += 1
                    print(f"✗ {model_name}: 未找到更新时间")

            return update_time

        except requests.RequestException as e:
            with self.progress_lock:
                self.failed_count += 1
                print(f"✗ {model_name}: 请求失败 - {e}")
            return None

    def update_csv_file(self, input_file: str, output_file: str) -> None:
        """
        更新CSV文件中的缺失数据
        """
        # 读取CSV文件
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # 找出需要更新的行
        models_to_update = []
        for i, row in enumerate(rows):
            if row['update_time'] == '-' or not row['update_time']:
                models_to_update.append((i, row['name']))

        print(f"发现 {len(models_to_update)} 个模型需要更新")

        # 使用线程池并行获取更新时间
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_model = {
                executor.submit(self.fetch_model_update_time, model_name): (index, model_name)
                for index, model_name in models_to_update
            }

            # 处理完成的任务
            for future in concurrent.futures.as_completed(future_to_model):
                index, model_name = future_to_model[future]
                try:
                    update_time = future.result()
                    if update_time:
                        rows[index]['update_time'] = update_time
                except Exception as e:
                    print(f"处理 {model_name} 时出错: {e}")

        # 写入更新后的数据
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            fieldnames = reader.fieldnames
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"\n更新完成！")
        print(f"成功更新: {self.updated_count} 个模型")
        print(f"更新失败: {self.failed_count} 个模型")
        print(f"输出文件: {output_file}")

def main():
    # 创建更新器实例（降低并发数，增加延迟）
    updater = HuggingFaceModelUpdater(max_workers=2, delay_range=(3, 5))

    # 输入和输出文件
    input_file = "qwen_models_cleaned.csv"
    output_file = "qwen_models_updated_v2.csv"

    print("开始更新Qwen模型数据...")
    print("注意：这个过程可能需要一些时间，请耐心等待")
    print("=" * 50)

    # 执行更新
    updater.update_csv_file(input_file, output_file)

if __name__ == "__main__":
    main()