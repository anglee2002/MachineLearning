# -*- coding: utf-8 -*- 
# @Time : 5/29/23 17:10 
# @Author : ANG

import openai
import pandas as pd

# 设置你的 OpenAI API 密钥
openai.api_key = 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'


# 读取 CSV 文件
def read_csv(dataset):
    data = pd.read_csv(dataset)
    # 截取 20% 的数据进行测试
    size = int(len(data) * 0.2)
    data = data.sample(n=size)
    return data


# 生成提示文本
def generate_prompt(text):
    """
    为了提高正确率，使用英文提示文本，并增加了一些提示文本（均来自comments.csv文件）
    该prompt已经在OpenAI Playground进行验证，能够输出格式话结果，便于后续的结果比对。
    """
    return """Please determine the sentiment polarity (positive or negative) of the following text:
text: 这款手机在使用上非常顺手
result: positive
text: {}
result:""".format(
        text.capitalize()
    )


# 情感语义分析
def analyze_sentiment(text):
    prompt = generate_prompt(text)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": generate_prompt(text)},
        ],
        max_tokens=20,
        temperature=0,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        n=1,
    )

    sentiment = response.choices[0].message.content.strip()

    return sentiment


# 分析情感并与正确结果比对正确率
def analyze_and_evaluate(dataset):
    data = read_csv(dataset)
    # 总行数
    total_count = len(data)
    # 正确行数
    correct_count = 0

    for index, row in data.iterrows():
        text = row['comments']
        expected_sentiment = int(row['label'])

        sentiment = analyze_sentiment(text)
        predicted_sentiment = 1 if sentiment == "positive" else 0

        if predicted_sentiment == expected_sentiment:
            correct_count += 1

    accuracy = correct_count / total_count * 100
    print("准确率：{:.2f}%".format(accuracy))


# 测试情感分析并评估准确率
dataset = '../数据集/comments.csv'
analyze_and_evaluate(dataset)
