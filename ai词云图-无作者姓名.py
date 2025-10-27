import requests
from bs4 import BeautifulSoup
import re
import jieba
from wordcloud import WordCloud
from pypinyin import lazy_pinyin
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from PIL import Image
import urllib3
import os
import json
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

COLOR_MAP = {
    "红色": "#E74C3C", "金色": "#FFD700", "蓝色": "#3498DB", "灰色": "#7F8C8D",
    "黑色": "#000000",  "绿色": "#27AE60",
    "紫色": "#8E44AD", "橙色": "#FFA500", "黄色": "#F1C40F", "青色": "#1ABC9C",
    "粉色": "#FFC0CB", "棕色": "#8B4513", "淡蓝色": "#ADD8E6", "深蓝色": "#003366",
    "深绿": "#006400", "淡黄": "#FFFACD"
}

def get_main_imagery(poet_name, min_freq=3, max_words=85):   #意象数量
    pinyin_result = ''.join(lazy_pinyin(poet_name))
    target_url = f"https://www.shiku.org/shiku/xs/{pinyin_result}.htm"
    project_dir = r"C:\Users\29176\Desktop\pku--学术\词云py"     #保存地址
    os.makedirs(project_dir, exist_ok=True)

    pronouns = {'你','我','他','她','它','我们','你们','他们','她们','它们'}
    function_words = {'不','这','那','那里','那个','一次','一条','那是','这是','就是','变成','那些','这里','突然','一种','一片','一只','一个','一起','没有','和','的','去','了','着','呢','吗','吧','嘛','啊','呀','哦','哎','呗','么','而','其','之','与','于','并','也','则','乃','且','尚','或','及','以','乎','若','焉','哉','所','为','被','把','将','在','向','于','从','到','由','给','对','比','把'}
    remove_words = pronouns | function_words

    merge_dict = {
        '太阳': '太阳', '月亮': '月亮', '春风': '风', '秋风': '风', '江水': '江水', '河流': '河流',
        '花朵': '花', '春天': '春', '夏天': '夏', '秋天': '秋', '冬天': '冬',
        '黑夜': '夜', '白昼': '白昼', '晨': '白昼', '清晨': '白昼', '朝阳': '太阳', '晚霞': '晚霞'
        #比较粗糙，可根据实际需要扩充
    }

    try:
        print(f"正在抓取: {target_url}")
        response = requests.get(target_url, verify=False, timeout=30)
        response.encoding = 'gbk'
        if response.status_code != 200 or "404 Not Found" in response.text:
            print(f"未找到诗人 {poet_name} 的专页")
            return {}, project_dir

        soup = BeautifulSoup(response.text, 'html.parser')
        all_elements = soup.find_all(True)
        start_index = len(all_elements) // 5
        poem_elements = all_elements[start_index:]

        poems = []
        for element in poem_elements:
            text = element.get_text().strip()
            if text and len(text) > 10 and re.search(r'[\u4e00-\u9fa5]{5,}', text):
                cleaned = re.sub(r'[^\u4e00-\u9fa5]', '', text)
                for w in remove_words:
                    cleaned = cleaned.replace(w, '')
                cleaned = re.sub(r'\s+', '', cleaned)
                poems.append(cleaned)

        all_text = ''.join(poems)
        txt_file_path = os.path.join(project_dir, f"{poet_name}_全部诗歌.txt")
        with open(txt_file_path, 'w', encoding='utf-8') as f:
            f.write(all_text)
        print(f"全部诗歌已保存到: {txt_file_path}")

        # 分词统计
        words = [w for w in jieba.cut(all_text) if len(w) > 1 and re.match(r'^[\u4e00-\u9fa5]+$', w)]
        # 合并意象
        merged_words = [merge_dict.get(w, w) for w in words]
        freq = Counter(merged_words)
        # 只保留高频意象
        imagery_freq = dict(sorted({k: v for k, v in freq.items() if v >= 2}.items(), key=lambda x: -x[1]))
        # 只取前max_words个
        main_imagery = dict(list(imagery_freq.items())[:max_words])
        print(f"筛选高频意象数：{len(main_imagery)}")
        return main_imagery, project_dir

    except Exception as e:
        print(f"抓取出错: {e}")
        return {}, project_dir

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))


def analyze_imagery_emotion_color(imagery_freq, poet_name, project_dir):
    client = OpenAI(
        api_key="sk-3d7ea3268a694836927d0e7247721aa2",  # 替换为你的API密钥
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout=30
    )
    imagery_list = list(imagery_freq.keys())
    prompt = f"""
以下是{poet_name}诗歌中的主要意象及其出现频率，请你为每一个意象提供对应颜色。以下是意象-颜色匹配的原则：
1．	其中对于具象的意象，请你提供其在日常生活中最常出现的颜色；
2．	对于抽象的意象，请你提供其对应的情绪，以及情绪对应的颜色词；
3．	对于没有情绪偏向的意象，请你随机提供其对应的颜色
对于每个意象，输出颜色的中文，且只能在'红色,金色,蓝色,灰色,黑色,绿色,紫色,橙色,黄色,青色,粉色,棕色,淡蓝色,深蓝色,深绿,淡黄'之中选择。
请你按照以上原则进行思考，写下思考过程，并最终提供意象和对应的颜色.
意象词及频率: {json.dumps(imagery_freq, ensure_ascii=False)};无需返回情绪词，只返回意象词和对应的颜色。
输出示例:
"""
    print("调用大模型分析意象的颜色...")
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1000,                   ##tokens数目
        response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    try:
        result = json.loads(content)
    except Exception as e:
        print("API返回内容解析失败:", e)
        result = {}
    # 保存json
    json_file = os.path.join(project_dir, f"{poet_name}_意象情绪颜色.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"意象情绪及颜色分析已保存: {json_file}")
    return result

def color_func_factory(emotion_color_dict):
    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        # 获取颜色映射，如果没有则使用默认灰色
        color = COLOR_MAP.get(emotion_color_dict.get(word, {}).get('color', ''), "#333333")
        return color
    return color_func

def generate_colored_wordcloud(imagery_freq, emotion_color_dict, poet_name, output_dir):
    # 创建词云对象
    wc = WordCloud(
        font_path='simkai.ttf',
        width=1600,
        height=1200,
        background_color='white',
        max_words=200,                        #
        prefer_horizontal=0.6,
        min_font_size=10,
        max_font_size=200,
        margin=2,
        scale=2,
        repeat=False,
        random_state=42
    )
    
    # 生成词云
    wc.generate_from_frequencies(imagery_freq)
    
    # 设置颜色函数
    wc.recolor(color_func=color_func_factory(emotion_color_dict))
    
    # 绘制词云
    plt.figure(figsize=(16, 12))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'{poet_name}诗歌意象词云(情绪颜色)', fontsize=24, pad=20)
    
    # 保存词云
    output_file = os.path.join(output_dir, f'{poet_name}_情绪颜色词云.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

if __name__ == "__main__":
    poet_name = input("请输入诗人名字: ").strip()
    
    # 设置输出目录
    output_dir = r"C:\Users\29176\Desktop\pku--学术\词云py"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n开始分析 {poet_name} 的诗歌意象...")
    imagery_freq, project_dir = get_main_imagery(poet_name)
    
    if imagery_freq:
        print(f"\n高频意象统计: {imagery_freq}")
        
        # 分析意象情绪和颜色
        emotion_color_dict = analyze_imagery_emotion_color(imagery_freq, poet_name, project_dir)
        print(f"\n意象情绪颜色分析结果: {emotion_color_dict}")
        
        # 生成彩色词云
        print("\n生成情绪颜色词云中...")
        output_file = generate_colored_wordcloud(imagery_freq, emotion_color_dict, poet_name, output_dir)
        
        if output_file:
            print(f"\n词云已保存: {output_file}")
            try:
                Image.open(output_file).show()
            except:
                print("请手动查看图片文件")

    else:
        print("\n获取失败，请检查：")
        print("- 诗人名称是否正确")
        print("- 网络连接是否正常")
        print("- 尝试其他诗人名称")