import argparse
import json
import random
import traceback
import fitz  # PyMuPDF
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
import os
import time
import re
import math
from collections import Counter
import requests
from scipy import stats 
import pyperclip
import spacy
import re
from openai import OpenAI
from tqdm import tqdm
import psutil
import subprocess

def add_date_to_logs():
    """为日志文件添加日期标记，以区分不同的处理轮次"""
    log_files = [
        '_error_pdfs.log',
        '_invalid_pdfs_case.log',
        '_processed_pdfs.log',
        '_processed_pdfs_requests_dump.log',
        '_task_duration.log',
        '_scanned_pdfs_case.log',
        '_skipped_pdfs.log'
    ]
    
    current_date = time.strftime("%Y-%m-%d %H:%M:%S")
    separator = f"\n{'='*50}\n新的处理轮次开始于: {current_date}\n{'='*50}\n"
    
    for log_file in log_files:
        if os.path.exists(log_file):
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(separator)
        else:
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"日志文件创建于: {current_date}\n{separator}")
    
    print(f"已为所有日志文件添加日期标记: {current_date}")

# 在程序开始时调用此函数
add_date_to_logs()

# 读取需要跳过的路径列表
skip_paths = []
try:
    with open('skip_path_list.txt', 'r', encoding='utf-8') as skip_file:
        skip_paths = [os.path.normpath(line.strip()) for line in skip_file]
    print(f"已读取跳过路径列表: {skip_paths[:10]}")
except FileNotFoundError:
    print("未找到skip_path_list.txt文件，将处理所有PDF")


include_paths = []
try:
    with open('include_path_list.txt', 'r', encoding='utf-8') as include_file:
        include_paths = [os.path.normpath(line.strip()) for line in include_file]
    print(f"已读取包含路径列表: {include_paths[:10]}")
except FileNotFoundError:
    print("未找到include_path_list.txt文件，将处理所有PDF")

# nlp = spacy.load("en_core_web_sm")


def has_chinese(text):
    pattern = re.compile(r'[\u4e00-\u9fff]')
    return pattern.search(text) is not None

def is_regular_content(text):
    words = re.findall(r'[a-zA-Z]{2,}', text)  # 提取至少包含2个字母的英文单词
    if len(words) >= 5:
        for i in range(len(words) - 4):
            if all(len(word) >= 2 for word in words[i:i+5]):
                return True
    return False
    # doc = nlp(text)
    # for token in doc:
    #     if token.pos_ in ["NOUN", "VERB", "ADJ"]:
    #         return True
    # # 检查文本是否包含常见的中文标点符号
    # if re.search(r'[，。！？、""'']', text):
    #     return True
    # # 检查文本是否以大写字母开头,并且包含多个单词
    # if re.match(r'^[A-Z][a-z]+ ', text):
    #     return True
    # # 检查文本是否包含常见的英文单词
    # common_words = ['the', 'and', 'is', 'are', 'of', 'in', 'to', 'for', 'on', 'with']
    # if any(word in text.lower() for word in common_words):
    #     return True
    # return False

import hashlib

def get_pdf_md5(pdf_path):
    hash_md5 = hashlib.md5()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def is_script_running(script_path):
    script_name = os.path.basename(script_path)
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Check if it's a Python process
            if process.info['name'].lower().startswith('python'):
                cmdline = process.info['cmdline']
                print(cmdline)
                # Check if the script name is in the command line arguments
                if len(cmdline) > 1 and script_name in cmdline[-1]:
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False
def translate_paragraphs(paragraphs, pdf_path):
    url = "http://localhost:8000/translate"
    
    # 分离常规内容和非常规内容
    regular_content = []
    non_regular_content = []
    for i, para in enumerate(paragraphs):
        if is_regular_content(para):
            regular_content.append((i, para))
        else:
            non_regular_content.append((i, para))
    
    # 只发送常规内容进行翻译
    data = {
        "paragraphs": [para for _, para in regular_content],
        "target_lang": "zh",
        "test_mode": False  # 设置为 True 来启用测试模式，返回原始段落
    }
    
    if dump_json_only:
        pdf_md5 = get_pdf_md5(pdf_path)
        key = f"{os.path.basename(pdf_path)}_{pdf_md5}"
        existing_data[key] = data
        translated_paragraphs = ["DUMP_JSON_MODE_ " + para for _, para in regular_content]
    else:
        response = requests.post(url, json=data)
        if response.status_code != 200:
            raise Exception(f"Translation failed: {response.text}")
        translated_paragraphs = response.json()["translated_paragraphs"]
    
    # 合并翻译结果和非常规内容
    result = [None] * len(paragraphs)
    for (i, _), translated in zip(regular_content, translated_paragraphs):
        result[i] = translated
    for i, para in non_regular_content:
        result[i] = para
    
    return result



import asyncio
import logging
from datetime import datetime
from openai import AsyncOpenAI

def translate_paragraphs_openai_stream(paragraphs, file_name):
    client = AsyncOpenAI(
        api_key="sk-......",
        base_url='https://sapi.onechats.top/v1/'
    )
    
    sem = asyncio.Semaphore(100)  # 限制并发数为100
    failure_counter = 0

    async def translate_paragraph(para, index):
        async with sem:
            for attempt in range(5):  # 最多重试5次
                try:
                    if is_regular_content(para):
                        # prompt = f"将以下文本翻译成中文：\n\n{para}\n\n翻译："                        
                        response = await client.chat.completions.create(
                            messages=[
                                {"role": "system", "content": "Translate the text into Chinese. ONLY RETURN TRANSLATED TEXT AND NOTHING ELSE."},
                                {"role": "user", "content": para}
                            ],
                            model="gpt-3.5-turbo",
                            stream=False
                        )
                        
                        translated_para = response.choices[0].message.content.strip()
                        print(f"\n已翻译段落 {index+1}/{len(paragraphs)}:\n{para[:60]}...\n{translated_para[:60]}...")
                        return translated_para
                    else:
                        return para
                except Exception as e:
                    current_time = datetime.now().strftime("%H:%M:%S")
                    print(f"{current_time} 尝试 {attempt+1}\n - 段落:\n {para[:40]}\n - 错误: {str(e)}\n")
                    logging.error(f"{current_time} 尝试 {attempt+1}\n - 段落:\n {para}\n - 错误: {str(e)}\n")
                    await asyncio.sleep(1)  # 等待1秒后重试
            
            nonlocal failure_counter
            failure_counter += 1
            with open(f'{file_name}_failed_paragraphs.log', 'a') as f:
                f.write(para + '\n\n')
            return None

    async def translate_all():
        tasks = [translate_paragraph(para, i) for i, para in enumerate(paragraphs)]
        return await asyncio.gather(*tasks)

    logging.basicConfig(filename=f'{file_name}_error.log', level=logging.ERROR)
    
    # 使用事件循环运行异步函数
    loop = asyncio.get_event_loop()
    translated_paragraphs = loop.run_until_complete(translate_all())
    
    print(f'翻译完成。失败次数: {failure_counter}。请查看日志文件了解详情。')
    return translated_paragraphs


def word_num(text):
    pattern = re.compile(r'[\u4e00-\u9fff]')
    if pattern.search(text) is not None:
        return len(pattern.findall(text))
    else:
        return len(text.split())

def is_scanned_pdf(pdf_path):
    print(f"Checking if {pdf_path} is a scanned PDF...")
    doc = fitz.open(pdf_path)
    # Check the first 10 pages
    for i in range(min(10, len(doc))):
        page = doc.load_page(i)
        if page.get_text():
            return False  # Not a scanned PDF
    return True  # No text found in the first 10 pages, probably a scanned PDF

def get_toc_info_per_page(pdf_path):
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()  # Get TOC information
    print(f"TOC: {toc}")

    page_toc_dict = {}
    if not toc:
        return page_toc_dict

    for page in range(1, doc.page_count + 1):
        for i in range(len(toc) - 1):
            current_item, next_item = toc[i], toc[i+1]
            if current_item[2] <= page <= next_item[2]:
                lvl, title, pagenum, *_ = current_item
                if page not in page_toc_dict:
                    page_toc_dict[page] = []
                page_toc_dict[page].append({'level': lvl, 'title': title.strip()})
                # break

        # For the last TOC item, assume it continues to the last page of the document
        if page >= toc[-1][2]:
            lvl, title, pagenum, *_ = toc[-1]
            if page not in page_toc_dict:
                page_toc_dict[page] = []
            page_toc_dict[page].append({'level': lvl, 'title': title.strip()})

    return page_toc_dict


def extract_paragraphs(pdf_path, heuristic):
    doc = fitz.open(pdf_path)
    
    pdfdata = doc.tobytes()
    doc_original = fitz.open("pdf", pdfdata)

    paragraphs = []
    hot_slot_page_num = ''
    hot_slot_index = ''
    line_number = 0
    page_nums = []
    toc_titles = []

    toc_info = get_toc_info_per_page(pdf_path)

    # for page, info in toc_info.items():
    #     print(f"Page {page} TOC Info: {info}")

    # Store all blocks in a list
    all_blocks_and_its_page = []
    # store all aspect ratio of blocks in a list
    all_blocks_aspect_ratio = []
    # List to store span heights
    span_heights = []
    first_word_of_the_span = []

    text_and_punctuations_before_indentation_dict = {}

    all_line_heights = [] 
    mean_line_height = 0
    
    from tqdm import tqdm
    
    pbar_pages = tqdm(enumerate(doc), total=len(doc), desc="正在提取文本块")
    for i, page in pbar_pages:
        text_and_punctuations_before_indentation_dict[i+1] = []
        # pbar_pages.set_description(f"提取第 {i+1} 页的文本块")
        pbar_pages.update(1)
        data = page.get_text("dict")
        # for block in data['blocks']:
        for j, block in enumerate(data['blocks']):
            if block['type'] == 0:
                # mark the boundary of the block
                # page.draw_rect(block['bbox'], color=(0, 1, 0), width=0.5)
                all_blocks_and_its_page.append((page, block, i+1, j))
                # calculate the aspect ratio of the block
                block_aspect_ratio = (block['bbox'][2] - block['bbox'][0])/(block['bbox'][3] - block['bbox'][1])
                all_blocks_aspect_ratio.append(block_aspect_ratio)
                for line in block["lines"]:
                    line_height = line['bbox'][3] - line['bbox'][1]
                    all_line_heights.append(line_height) 
                    for span in line["spans"]:
                        # Add the height of the span to the list
                        rect = fitz.Rect(span["bbox"])
                        height = math.ceil(rect.height)  # Round height up to the nearest integer
                        span_heights.append(height)
                        first_word_of_the_span.append([span['text'][:20],height])
    
    mean_line_height = np.mean(all_line_heights)
    # Count occurrences of each span height
    span_height_counter = Counter(span_heights)
    # Get the most common span height
    span_height = span_height_counter.most_common(1)[0][0]
    print("The most common rounded-up span height is:", span_height)



    paragraphs_to_be_translated_list = []
    paragraphs_to_be_translated_bbox_list = []
    paragraphs_to_be_translated_page_num_list = []

    current_paragraph = ''
    # create a list obj that can be assigned to a variable
    current_block_bbox = [0,0,0,0]
    header_detected_dict = {}

    from tqdm import tqdm
    
    pbar = tqdm(total=len(all_blocks_and_its_page), desc="处理文本块", colour="green")
    for i, (page, block, page_number, block_index) in enumerate(all_blocks_and_its_page):
        # pbar.set_description(f"处理第 {page_number} 页的第 {block_index + 1} 个文本块")
        pbar.update(1)
        last_line_right = None
        last_line_left = None
        merged_line = {}
        block_number = i + 1
        # check if header detected dict has key page number
        if page_number not in header_detected_dict:
            header_detected_dict[page_number] = False

        top_right = (block['bbox'][2] - 15, block['bbox'][1] + 5)  # top-right coordinates of the block
        page.insert_text(top_right, str(block_number), fontsize=5, color=(1, 0, 0))  # insert block number

        full_page_bbox = page.rect
        upper_left_quarter = fitz.Rect(full_page_bbox.x0, full_page_bbox.y0, full_page_bbox.width / 2, full_page_bbox.height / 2)
        lower_right_corner = fitz.Rect(full_page_bbox.width / 2, full_page_bbox.height * 3/4, full_page_bbox.width, full_page_bbox.height)

        # draw a 5 pixel square at the upper left corner of the page, and 10 pixel on the upper right corner
        page.draw_rect((0, 0, 5, 5), color=(1, 0, 0), fill=(1, 0, 0), width=0.5)
        page.draw_rect((full_page_bbox.width - 10, 0, full_page_bbox.width, 10), color=(1, 0, 0), fill=(1, 0, 0), width=0.5)

        for line in block['lines']:
            # Merge spans on the same line                    
            if merged_line and abs(merged_line['bbox'][1] - line['bbox'][1]) < 0.25 * span_height \
                and line['bbox'][0] - merged_line['bbox'][2] > 0.1 * span_height:
                merged_line['bbox'][2] = max(merged_line['bbox'][2], line['bbox'][2])
                merged_line['bbox'][3] = max(merged_line['bbox'][3], line['bbox'][3])
                merged_line['text'] += ' ' + ' '.join([span['text'] for span in line['spans']])

            else:
                # Process the merged line
                if merged_line:
                    line_text = merged_line['text']
                    current_line_left, _, current_line_right, _ = merged_line['bbox']

                    if current_paragraph:
                        is_indented = last_line_left is not None and current_line_left - last_line_left > 0.7 * span_height
                        line_not_filled = last_line_right is not None and current_line_right - last_line_right > 4 * span_height
                        # Check if the last line ends with punctuation
                        if current_paragraph.strip():
                            last_line_ends_with_punctuation = current_paragraph.strip()[-1] in ['.', '!', '?', '。', '！', '？','"', '”']
                        else:
                            last_line_ends_with_punctuation = False
                        last_line_ends_with_punctuation_and_script = re.search(r'[.?!。！？”"] *\d+$', current_paragraph.strip()) is not None
                        last_line_ends_with_punctuation = last_line_ends_with_punctuation or last_line_ends_with_punctuation_and_script
                        if (is_indented or line_not_filled) and last_line_ends_with_punctuation:

                            text_and_punctuations_before_indentation_dict[page_number].append(current_paragraph[-10:].strip())

                            # check if the upper left corner of the block is in the upper left quarter of the page
                            # if block['bbox'][0] < upper_left_quarter.x1 and block['bbox'][1] < upper_left_quarter.y1:
                            #     first_line_in_upper_left_quarter = True
                            # else:
                            #     first_line_in_upper_left_quarter = False
                            # current_paragraph start with a lower case letter
                            if 000000 and (current_paragraph[0].islower() or has_chinese(current_paragraph)) and \
                                word_num(current_paragraph) > 20 and hot_slot_page_num == page_number -1:
                                # and first_line_in_upper_left_quarter # too hash
                                # current_paragraph is a continuation of the last paragraph
                                last_2_words = ' '.join(paragraphs[hot_slot_index].split()[-2:])
                                page.insert_text((0 + 5, block['bbox'][1] + 5), last_2_words, fontsize=5, color=(0, 0, 0))
                                paragraphs[hot_slot_index] += ' ' + current_paragraph
                                hot_slot_page_num = ''
                                hot_slot_index = ''
                                page.draw_rect((block['bbox'][0] - 7, block['bbox'][1], block['bbox'][0] - 5, block['bbox'][3]), color=(0, 1, 0), fill = (0, 1, 0), width=0.5)
                                page.insert_text((block['bbox'][0] - 7, block['bbox'][1] + 5), 'continued_1', fontsize=5, color=(0, 0, 0))
                                page.draw_rect((block['bbox'][2], merged_line['bbox'][1] - 2, block['bbox'][2] + 2, merged_line['bbox'][1]), color=(0, 0, 0), fill = (0,0,0), width=0.5)
                                page.insert_text((block['bbox'][2] + 2, merged_line['bbox'][1]), 'by_new_para', fontsize=5, color=(0, 0, 0))
                            else:
                                paragraphs.append(current_paragraph)
                                # 对文本块进行修订并插入模拟翻译
                                bbox = current_block_bbox
                                rect = fitz.Rect(bbox)
                                # # reversed_text = current_paragraph[::-1]  # 反转文本作为模拟翻译
                                # # reversed_text = reversed_text.upper()
                                # reversed_text = '> ' + current_paragraph + ' <'
                                # # 使用红色注释覆盖原文本
                                # page.add_redact_annot(rect, text="")
                                # page.apply_redactions()
                                # # 插入模拟翻译后的文本,红色字体
                                # html = f'''
                                #     {reversed_text}
                                # '''
                                # page.insert_htmlbox(rect, html, css="* {background-color: red; font-size: 30px;}")
                                # page.draw_rect(rect, color=(0, 1, 0), width=2)

                                # 将文本添加到待翻译列表
                                paragraphs_to_be_translated_list.append(current_paragraph)
                                paragraphs_to_be_translated_bbox_list.append(rect)
                                paragraphs_to_be_translated_page_num_list.append(page_number)



                                page_nums.append(page_number)
                                page.draw_rect((block['bbox'][2], merged_line['bbox'][1] - 2, block['bbox'][2] + 2, merged_line['bbox'][1]), color=(0, 0, 0), fill = (0,0,0), width=0.5)
                                page.insert_text((block['bbox'][2] + 2, merged_line['bbox'][1]), 'by_new_para', fontsize=5, color=(0, 0, 0))
                                titles_list = []
                                for info in toc_info.get(page_number, [{'title': 'N/A'}]):
                                    titles_list.append(info['title'])
                                title_concat = ' | '.join(titles_list)
                                toc_titles.append(title_concat)

                                # toc_titles.append(toc_info.get(page_number, [{'title': 'N/A'}])[0]['title'])
                            current_paragraph = line_text
                            current_block_bbox = list(merged_line['bbox'])
                            page.draw_rect(merged_line['bbox'], color=(0, 1, 0), width=0.5)
                        else:
                            current_paragraph += ' ' + line_text
                            current_block_bbox[0] = min(current_block_bbox[0], merged_line['bbox'][0])
                            current_block_bbox[2] = max(current_block_bbox[2], merged_line['bbox'][2])
                            current_block_bbox[3] = max(current_block_bbox[3], merged_line['bbox'][3])
                            # Normal extended line, draw a black box
                            page.draw_rect(merged_line['bbox'], color=(0, 0, 0), width=0.5)                                    
                    else:
                        current_paragraph = line_text
                        current_block_bbox = list(merged_line['bbox'])

                    last_line_right = current_line_right
                    last_line_left = current_line_left

                    
                # Start a new merged line
                merged_line = {
                    'bbox': list(line['bbox']),
                    'text': ' '.join([span['text'] for span in line['spans']]),
                }
                merged_line['bbox'][0] = min(merged_line['bbox'][0], block['bbox'][0])

                
                if 'small in relation to free-energy gradients' in merged_line['text']:
                    pass


                # --- 计算行高百分比排名并插入信息 ---
                line_height = merged_line['bbox'][3] - merged_line['bbox'][1]
                # 计算相比平均行高的倍数
                ratio_over_mena_line_height = line_height / mean_line_height
                line_height_percentile = stats.percentileofscore(all_line_heights, line_height)
                page.insert_text((merged_line['bbox'][0] - 20, (merged_line['bbox'][1] + merged_line['bbox'][3]) / 2),
                                f"{line_height_percentile:.0f}% ({ratio_over_mena_line_height:.1f})",
                                fontsize=5, color=(1, 0, 0))
                # page.insert_text((merged_line['bbox'][0] - 7, merged_line['bbox'][1] + 4), str(line_number), fontsize=5, color=(0, 0, 0))
                # line_number += 1

        # if '432 Proceedings of the IEEE' in merged_line['text']:
        #     pass

        # Don't forget to process the last merged line
        if merged_line:
            line_text = merged_line['text']
            current_line_left, _, current_line_right, _ = merged_line['bbox']
            
            if current_paragraph:
                is_indented = last_line_left is not None and current_line_left - last_line_left > 0.7 * span_height
                line_not_filled = last_line_right is not None and current_line_right - last_line_right > 4 * span_height
                # Check if the last line ends with punctuation
                if current_paragraph.strip():
                    last_line_ends_with_punctuation = current_paragraph.strip()[-1] in ['.', '!', '?', '。', '！', '？','"', '”']
                else:
                    last_line_ends_with_punctuation = False
                last_line_ends_with_punctuation_and_script = re.search(r'[.?!。！？”"] *\d+$', current_paragraph.strip()) is not None
                last_line_ends_with_punctuation = last_line_ends_with_punctuation or last_line_ends_with_punctuation_and_script

                if (is_indented or line_not_filled) and last_line_ends_with_punctuation:
                    text_and_punctuations_before_indentation_dict[page_number].append(current_paragraph[-10:].strip())
                    # check if the upper left corner of the block is in the upper left quarter of the page
                    # if block['bbox'][0] < upper_left_quarter.x1 and block['bbox'][1] < upper_left_quarter.y1:
                    #     first_line_in_upper_left_quarter = True
                    # else:
                    #     first_line_in_upper_left_quarter = False
                    # current_paragraph start with a lower case letter
                    if 000000 and (current_paragraph[0].islower() or has_chinese(current_paragraph)) and \
                        word_num(current_paragraph) > 20 and hot_slot_page_num == page_number -1:
                        #  and first_line_in_upper_left_quarter # too hash
                        # current_paragraph is a continuation of the last paragraph
                        last_2_words = ' '.join(paragraphs[hot_slot_index].split()[-2:])
                        page.insert_text((0 + 5, block['bbox'][1] + 5), last_2_words, fontsize=5, color=(0, 0, 0))
                        paragraphs[hot_slot_index] += ' ' + current_paragraph
                        hot_slot_page_num = ''
                        hot_slot_index = ''
                        page.draw_rect((block['bbox'][0] - 7, block['bbox'][1], block['bbox'][0] - 5, block['bbox'][3]), color=(0, 1, 0), fill = (0, 1, 0), width=0.5)
                        page.insert_text((block['bbox'][0] - 7, block['bbox'][1] + 5), 'continued_2', fontsize=5, color=(0, 0, 0))
                        page.draw_rect((block['bbox'][2], block['bbox'][3] - 4, block['bbox'][2] + 4, block['bbox'][3]), color=(0, 0, 0), fill = (0,0,0), width=0.5)
                        page.insert_text((block['bbox'][2], block['bbox'][3] - 5), 'by_last_line_new_para', fontsize=5, color=(0, 0, 0))# toc_titles.append(toc_info.get(page_number, [{'title': 'N/A'}])[0]['title'])
                    else:
                        paragraphs.append(current_paragraph)
                        # 对文本块进行修订并插入模拟翻译
                        bbox = current_block_bbox
                        rect = fitz.Rect(bbox)
                        # # reversed_text = current_paragraph[::-1]  # 反转文本作为模拟翻译
                        # # reversed_text = reversed_text.upper()
                        # reversed_text = '> ' + current_paragraph + ' <'
                        # # 使用红色注释覆盖原文本
                        # page.add_redact_annot(rect, text="")
                        # page.apply_redactions()
                        # # 插入模拟翻译后的文本,红色字体
                        # html = f'''
                        #     {reversed_text}
                        # '''
                        # page.insert_htmlbox(rect, html, css="* {background-color: red; font-size: 30px;}")
                        # page.draw_rect(rect, color=(0, 1, 0), width=2)

                        # 将文本添加到待翻译列表
                        paragraphs_to_be_translated_list.append(current_paragraph)
                        paragraphs_to_be_translated_bbox_list.append(rect)
                        paragraphs_to_be_translated_page_num_list.append(page_number)

                        page_nums.append(page_number)
                        page.draw_rect((block['bbox'][2], merged_line['bbox'][1] - 2, block['bbox'][2] + 2, merged_line['bbox'][1]), color=(0, 0, 0), fill = (0,0,0), width=0.5)
                        page.insert_text((block['bbox'][2] + 2, merged_line['bbox'][1]), 'by_last_line_new_para', fontsize=5, color=(0, 0, 0))# toc_titles.append(toc_info.get(page_number, [{'title': 'N/A'}])[0]['title'])
                        titles_list = []
                        for info in toc_info.get(page_number, [{'title': 'N/A'}]):
                            titles_list.append(info['title'])
                        title_concat = ' | '.join(titles_list)
                        toc_titles.append(title_concat)
                    current_paragraph = line_text
                    current_block_bbox = list(merged_line['bbox'])
                else:
                    current_paragraph += ' ' + line_text
                    current_block_bbox[0] = min(current_block_bbox[0], merged_line['bbox'][0])
                    current_block_bbox[2] = max(current_block_bbox[2], merged_line['bbox'][2])
                    current_block_bbox[3] = max(current_block_bbox[3], merged_line['bbox'][3])
            else:
                current_paragraph = line_text
                current_block_bbox = list(merged_line['bbox'])
        
        # look ahead for 未完待续 的迹象
        last_line_ends_with_punctuation = False
        if current_paragraph.strip():
            last_line_ends_with_punctuation = current_paragraph.strip()[-1] in ['.', '!', '?', '。', '！', '？','"', '”']
        last_line_ends_with_punctuation_and_script = re.search(r'[.?!。！？”"] *\d+$', current_paragraph.strip()) is not None
        last_line_ends_with_punctuation = last_line_ends_with_punctuation or last_line_ends_with_punctuation_and_script

        unfinished_column = False
        just_first_line = False
        just_sparsed_line = False
        just_left_overhang = False

        # if i + 2 < len(all_blocks_and_its_page):
        if i + 1 < len(all_blocks_and_its_page):
            # 是否未完待续

            next_block = all_blocks_and_its_page[i + 1][1]
            # next_block_2 = all_blocks_and_its_page[i + 2][1]
            # 间隙蓝色方框
            # also skip appending if this block is left and right-aligned with the next block and 
            # not ending with punctuation which means in a large chance it is a wrongly classified regular line in a block
            left_indentation = block['bbox'][0] - next_block['bbox'][0]
            left_overhang = next_block['bbox'][0] - block['bbox'][0]
            horizontal_gap = next_block['bbox'][0] - block['bbox'][2]
            # horizontal_gap_2 = next_block_2['bbox'][0] - block['bbox'][2]
            vertical_gap = next_block['bbox'][1] - block['bbox'][3]
            left_alignment = abs(block['bbox'][0] - next_block['bbox'][0])
            right_alignment = abs(block['bbox'][2] - next_block['bbox'][2])
            block_height = block['bbox'][3] - block['bbox'][1]
            # draw a rect around the spacing between two blocks
            page.draw_rect((block['bbox'][0], block['bbox'][3], next_block['bbox'][2], next_block['bbox'][1]), color=(0, 0, 1), width=0.5)
            # insert text to mark the spacing between two blocks with round to 2 decimal places
            text_ = str(round(left_alignment, 2))+' - '+str(round(right_alignment, 2))+' - '+str(round(vertical_gap, 2))
            page.insert_text((block['bbox'][0] + 5, block['bbox'][3] + 5), text_, fontsize=5, color=(0, 0, 0))

            # 右侧蓝色方框
            # 未完的左侧 Column
            # 有待继续
            # Skip appending if the paragraph is not finished on this block
            # vertical_alignment = abs(block['bbox'][1] - next_block['bbox'][1])
            # horizontal_gap = next_block['bbox'][0] - block['bbox'][2]
            if current_paragraph and 4 * span_height > horizontal_gap > span_height \
                and not last_line_ends_with_punctuation:
                # draw rect at the end of the paragraph
                page.draw_rect((block['bbox'][2] + 3, block['bbox'][1], block['bbox'][2] + 5, block['bbox'][3]), color=(1, 0, 0), fill=(1, 0, 0), width=0.5)
                page.insert_text((block['bbox'][2] + 5, block['bbox'][1] + 5), 'unfinished_column', fontsize=5, color=(0, 0, 0))
                # forced to be false as delayed redact could be complex
                unfinished_column = False
                # continue
            
            # 左侧红色方框
            # 实为自然段首行
            # 有待继续
            # Also skip appending if this block is a wrongly classified first line of the next block
            # right_alignment = abs(block['bbox'][2] - next_block['bbox'][2])
            # left_indentation = block['bbox'][0] - next_block['bbox'][0]
            # block_height = block['bbox'][3] - block['bbox'][1]
            if right_alignment < span_height and 5 * span_height > left_indentation > 0.7 * span_height and vertical_gap < 0.5 * span_height and block_height < 2 * span_height:
                # This block is a wrongly classified first line of the next block
                page.draw_rect((block['bbox'][2] + 1, block['bbox'][1], block['bbox'][2] + 3, block['bbox'][3]), color=(1, 0, 0), width=0.5)
                page.insert_text((block['bbox'][2] + 3, block['bbox'][1] + 5), 'just_first_line', fontsize=5, color=(0, 0, 0))
                just_first_line = True
                # continue
            

            # 左侧红色方框
            # 实为行间距较大的普通行
            # 有待继续
            # to deal with the large line spacing and thus wrongly classified regular line in a block, now consider the punctuation in this end
            # if left_alignment < 5 and right_alignment < 5 and vertical_gap < 5:
            # not even consider right alignment to include the last line of a paragraph that would normally not filled with text
            # block_height = block['bbox'][3] - block['bbox'][1]

            # if left_alignment < 0.7 * span_height and \
            #     (right_alignment < 0.7 * span_height or block['bbox'][2] - next_block['bbox'][2] > 0) and \
                # vertical_gap < 0.5 * span_height: # 2023-10-2

                # right_alignment < 0.7 * span_height and \ 2023-10-20
            if left_alignment < 0.7 * span_height and \
                0 < vertical_gap < 0.5 * span_height and block_height < 2 * span_height:
                # vertical_gap < 0.5 * span_height and block_height < 2 * span_height: as some two or more lines are recognized as one
                # only when first line of next block wasn't indentated
                left_of_first_line_of_next_block = next_block['lines'][0]['bbox'][0]
                if left_of_first_line_of_next_block - block['bbox'][0] < span_height:
                    line_height = merged_line['bbox'][3] - merged_line['bbox'][1]
                    line_height_percentile = stats.percentileofscore(all_line_heights, line_height)
                    # ratio_over_mena_line_height = line_height / mean_line_height
                    if not line_height_percentile > 80 and not line_height > 1.5 * mean_line_height:
                        page.draw_rect((block['bbox'][2] - 1, block['bbox'][1], block['bbox'][2] + 1, block['bbox'][3]), color=(1, 0, 0), width=0.5)
                        page.insert_text((block['bbox'][2] + 1, block['bbox'][1] + 5), 'just_sparsed_line', fontsize=5, color=(0, 0, 0))
                        # whenever we are here, hot slot flag should not apply any more 
                        # hot_slot_page_num = ''
                        # hot_slot_index = ''
                        # no, not really
                        just_sparsed_line = True
                        # continue

            # left_overhang = next_block['bbox'][0] - block['bbox'][0]
            last_line_ends_with_punctuation = False
            if current_paragraph.strip():
                last_line_ends_with_punctuation = current_paragraph.strip()[-1] in ['.', '!', '?', '。', '！', '？','"', '”']
            last_line_ends_with_punctuation_and_script = re.search(r'[.?!。！？”"] *\d+$', current_paragraph.strip()) is not None
            last_line_ends_with_punctuation = last_line_ends_with_punctuation or last_line_ends_with_punctuation_and_script

            # num_last_chars = min(len(current_paragraph.strip()), 5)
            # current_paragraph_last_chars = current_paragraph.strip()[-num_last_chars:]
            # block_height = block['bbox'][3] - block['bbox'][1]
            if left_overhang > 0.7 * span_height and right_alignment < 0.7 * span_height and block_height < 2 * span_height \
                and vertical_gap < 0.5 * span_height and not last_line_ends_with_punctuation:
                page.draw_rect((block['bbox'][2] - 1, block['bbox'][1], block['bbox'][2] + 1, block['bbox'][3]), color=(1, 0, 0), width=0.5)
                page.insert_text((block['bbox'][2] + 1, block['bbox'][1] + 5), 'just_left_overhang', fontsize=5, color=(0, 0, 0))
                # page.insert_text((block['bbox'][2] + 1, block['bbox'][1] + 10), ''+current_paragraph_last_chars, fontsize=5, color=(0, 0, 0))
                # page.insert_text((block['bbox'][2] + 1, block['bbox'][1] + 15), str(len(current_paragraph.strip())), fontsize=5, color=(0, 0, 0))
                # whenever we are here, hot slot flag should not apply any more 
                # hot_slot_page_num = ''
                # hot_slot_index = ''
                # no, not really
                just_left_overhang = True
                # continue

        if current_paragraph:
            # check if the upper left corner of the block is in the upper left quarter of the page
            # if block['bbox'][0] < upper_left_quarter.x1 and block['bbox'][1] < upper_left_quarter.y1:
            #     first_line_in_upper_left_quarter = True
            # else:
            #     first_line_in_upper_left_quarter = False
            # also make sure it's not a header by checking vertical spacing between this block and the next block
            is_header = False   
            # aspect_ration = all_blocks_aspect_ratio[i]
            if block_index == 0 and i + 1 < len(all_blocks_and_its_page) and header_detected_dict[page_number] == False:
                # Skip appending if the paragraph is not finished on this block
                next_block = all_blocks_and_its_page[i + 1][1]
                left_alignment = abs(block['bbox'][0] - next_block['bbox'][0])
                right_alignment = abs(block['bbox'][2] - next_block['bbox'][2])
                vertical_gap = next_block['bbox'][1] - block['bbox'][3]
                block_height = block['bbox'][3] - block['bbox'][1]
                if block_height < 2 * span_height and vertical_gap > span_height:
                    page.draw_rect((block['bbox'][2], block['bbox'][1], block['bbox'][2] + 4, block['bbox'][1] + 4), fill=(1, 1, 0), width=0.5)
                    is_header = True
                    header_detected_dict[page_number] = True

                # and hot_slot_page_num == page_number -1 and first_line_in_upper_left_quarter \ # maybe too hash
            if 000000 and (not is_header) and (current_paragraph[0].islower() or has_chinese(current_paragraph)) and \
                word_num(current_paragraph) > 20 and hot_slot_page_num == page_number -1 \
                and not unfinished_column and not just_first_line and not just_sparsed_line and not just_left_overhang:

                text_and_punctuations_before_indentation_dict[page_number].append(current_paragraph[-10:].strip())

                # current_paragraph is a continuation of the last paragraph
                last_2_words = ' '.join(paragraphs[hot_slot_index].split()[-2:])
                page.insert_text((0 + 5, block['bbox'][1] + 5), last_2_words, fontsize=5, color=(0, 0, 0))
                paragraphs[hot_slot_index] += ' ' + current_paragraph
                hot_slot_page_num = ''
                hot_slot_index = ''
                page.draw_rect((block['bbox'][0] - 7, block['bbox'][1], block['bbox'][0] - 5, block['bbox'][3]), color=(0, 1, 0), fill = (0, 1, 0), width=0.5)
                page.insert_text((block['bbox'][0] - 7, block['bbox'][1] + 5), 'continued_3', fontsize=5, color=(0, 0, 0))
                page.draw_rect((block['bbox'][2], block['bbox'][3] - 4, block['bbox'][2] + 4, block['bbox'][3]), color=(0, 0, 0), fill = (0,0,0), width=0.5)
                page.insert_text((block['bbox'][2], block['bbox'][3] - 5), 'para_end', fontsize=5, color=(0, 0, 0))# toc_titles.append(toc_info.get(page_number, [{'title': 'N/A'}])[0]['title'])
                current_paragraph = ''

            elif not unfinished_column and not just_first_line and not just_sparsed_line and not just_left_overhang:

                text_and_punctuations_before_indentation_dict[page_number].append(current_paragraph[-10:].strip())

                paragraphs.append(current_paragraph)
                # 对文本块进行修订并插入模拟翻译
                bbox = current_block_bbox
                rect = fitz.Rect(bbox)
                # # reversed_text = current_paragraph[::-1]  # 反转文本作为模拟翻译
                # # reversed_text = reversed_text.upper()
                # reversed_text = '> ' + current_paragraph + ' <'
                # # 使用红色注释覆盖原文本
                # page.add_redact_annot(rect, text="")
                # page.apply_redactions()
                # # 插入模拟翻译后的文本,红色字体
                # html = f'''
                #     {reversed_text}
                # '''
                # # font-size 30
                # page.insert_htmlbox(rect, html, css="* {background-color: red; font-size: 30px;}")
                # page.draw_rect(rect, color=(0, 1, 0), width=2)

                # 将文本添加到待翻译列表
                paragraphs_to_be_translated_list.append(current_paragraph)
                paragraphs_to_be_translated_bbox_list.append(rect)
                paragraphs_to_be_translated_page_num_list.append(page_number)

                page_nums.append(page_number)                
                page.draw_rect((block['bbox'][2], block['bbox'][3] - 4, block['bbox'][2] + 4, block['bbox'][3]), color=(0, 0, 0), fill = (0,0,0), width=0.5)
                page.insert_text((block['bbox'][2], block['bbox'][3] - 5), 'para_end', fontsize=5, color=(0, 0, 0))# toc_titles.append(toc_info.get(page_number, [{'title': 'N/A'}])[0]['title'])
                # toc_titles.append(toc_info.get(page_number, [{'title': 'N/A'}])[0]['title'])
                titles_list = []
                for info in toc_info.get(page_number, [{'title': 'N/A'}]):
                    titles_list.append(info['title'])
                title_concat = ' | '.join(titles_list)
                toc_titles.append(title_concat)
                
                # set hot slot index and page number as a flag if current paragraph is not ended with punctuation
                # if current_paragraph.strip()[-1] not in ['.', '!', '?','。','！','？','"', '”']:
                # if end with lower case letter, set it as hot slot
                # check x1 of the last line of the block is almost the same as x1 of the block
                line_bboxs =  [line['bbox'] for line in block['lines']]
                # check if the lower right corner of the block is in the lower right quarter of the page
                if block['bbox'][2] > lower_right_corner.x0 and block['bbox'][3] > lower_right_corner.y0:
                    last_line_in_lower_right_corner = True
                else:
                    last_line_in_lower_right_corner = False
                # if last_line_ends_with_lower_case and last_line_in_lower_right_corner:
                if not last_line_ends_with_punctuation and block['bbox'][2] - line_bboxs[-1][2] < 3 * span_height and last_line_in_lower_right_corner:
                    if word_num(current_paragraph) > 20:
                        page.draw_rect((block['bbox'][2] + 5, block['bbox'][1], block['bbox'][2] + 7, block['bbox'][3]), color=(1, 0, 0), fill = (1, 0, 0), width=0.5)
                        page.insert_text((block['bbox'][2] + 7, block['bbox'][1] + 5), 'unfinished', fontsize=5, color=(0, 0, 0))
                        hot_slot_index = len(paragraphs) - 1
                        hot_slot_page_num = page_number
                    # check word count of last hot slot paragraph, update it if current paragraph is longer
                    # elif hot_slot_index != '' and len(paragraphs[hot_slot_index].split()) < word_num(current_paragraph):
                    #     page.draw_rect((block['bbox'][2], block['bbox'][1], block['bbox'][2] + 5, block['bbox'][3]), color=(0, 0, 1), fill = (0,0,1), width=0.5)
                    #     hot_slot_index = len(paragraphs) - 1
                    #     hot_slot_page = page_number
                current_paragraph = ''
                current_block_bbox = [0,0,0,0]



    if heuristic:

        paragraphs = []
        paragraphs_fixed = []

        with open(pdf_path[:-4] + '_paragraphs_fixed.txt', 'a') as f:
            f.write(pdf_path[:-4] + '_paragraphs_fixed:\n\n')

        doc_ = fitz.open(pdf_path)
        for i, page in enumerate(doc_):
            print(f"\n\nExtract paragraphs from page {i+1}...\n")
            text = page.get_text('text')

            # paragraph_break = re.compile(r'(?<=[.?!。！？”"] *\d+)\s*\n')
            paragraph_break = re.compile(r'(?<=[.?!。！？”"])\s*\n')
            page_paragraphs = re.split(paragraph_break, text)

            text_and_punctuations_before_indentation_on_this_page_list = text_and_punctuations_before_indentation_dict.get(i+1, [])
            # print(text_and_punctuations_before_indentation_on_this_page_list)
            text_and_punctuations_before_indentation_on_this_page_list = [text_and_punctuations.replace(' ', '') for text_and_punctuations in text_and_punctuations_before_indentation_on_this_page_list]
            # print(text_and_punctuations_before_indentation_on_this_page_list)

            # for text_and_punctuations in text_and_punctuations_before_indentation_on_this_page_list:
            #     if text_and_punctuations:
            #         text_and_punctuations_pattern = re.compile(re.escape(text_and_punctuations) + r'$')
            #         for paragraph in page_paragraphs:
            #             if not text_and_punctuations_pattern.search(paragraph):
            #                 print(paragraph)                    
            #                 text = text.replace(text_and_punctuations, text_and_punctuations + '|||||||||||||||||||')
            

            for paragraph in page_paragraphs:
                paragraph_ = paragraph.replace('\n', '')
                paragraph_ = paragraph_.replace(' ', '')
                match_any_text_and_punctuations = False
                for text_and_punctuations in text_and_punctuations_before_indentation_on_this_page_list:
                    text_and_punctuations_pattern = re.compile(re.escape(text_and_punctuations) + r'$')
                    if text_and_punctuations_pattern.search(paragraph_):
                        match_any_text_and_punctuations = True

                if not match_any_text_and_punctuations:
                    print('\nFalse positive paragraph_:')
                    print(text_and_punctuations_before_indentation_on_this_page_list)
                    print(paragraph_)
                    text = text.replace(paragraph, paragraph + '<||--||>')
                    with open(pdf_path[:-4] + '_paragraphs_fixed.txt', 'a') as f:
                        f.write(f'Page {i+1} paragraphs_fixed:\n{paragraph}\n\n')

            # paragraph_break = re.compile(r'(?<=[.?!。！？”"] *\d+)\s*\n')
            paragraph_break = re.compile(r'(?<=[.?!。！？”"])\s*\n')
            page_paragraphs = re.split(paragraph_break, text)
            for paragraph in page_paragraphs:
                if paragraph.strip():

                    paragraph = paragraph.replace(' ', '')
                    paragraph = paragraph.replace('<||--||>', '')

                    # split lines in the same paragraph
                    lines = paragraph.split('\n')
                    # join them back with a space if in english
                    if not has_chinese(paragraph):
                        paragraph = ' '.join(lines)
                    else:
                        paragraph = ''.join(lines)
                    paragraphs.append(paragraph)
                    page_nums.append(i+1)
                    titles_list = []
                    for info in toc_info.get(i+1, [{'title': 'N/A'}]):
                        titles_list.append(info['title'])
                    title_concat = ' | '.join(titles_list)
                    toc_titles.append(title_concat)
        
        # print(text_and_punctuations_before_indentation_dict)
        return paragraphs, page_nums, toc_titles, doc, []
    
    # doc.save(pdf_path[:-4] + "_marked.pdf")
    # doc.save(pdf_path[:25] + "_layout_marked.pdf")

    # print(span_heights)
    # print(span_height)
    # print(first_word_of_the_span)

    paragraphs_ = []
    for paragraph in paragraphs:

        if has_chinese(paragraph):
            paragraph = paragraph.replace(' ', '')

        paragraphs_.append(paragraph)

    
    ###############################


    print('等待翻译完成...')
    if use_openai:
        file_name = os.path.splitext(os.path.basename(pdf_path))[0]
        paragraphs_translation_list = translate_paragraphs_openai_stream(paragraphs_to_be_translated_list, file_name)
        print('使用OpenAI完成翻译！')
    else:
        # 添加30秒超时等待本地服务器启动
        start_time = time.time()
        retry_count = 0
        while True:
            try:
                paragraphs_translation_list = translate_paragraphs(paragraphs_to_be_translated_list, pdf_path)
                print('翻译完成！')
                break
            except requests.exceptions.ConnectionError:
                if time.time() - start_time > 30:
                    raise Exception("本地翻译服务器30秒内未能启动，请检查服务器状态。")
                retry_count += 1
                print(f"正在尝试连接本地翻译服务器，第 {retry_count} 次重试...")
                time.sleep(1)
    
    if dump_json_only:
        return
                            
    print("翻译结果:")
    for i, translation in enumerate(paragraphs_translation_list, 1):
        print(f"段落 {i}: {translation}")
        # 模拟翻译结果,生成与原文单词数相近的中文
        # paragraphs_translation_list = []
        # for paragraph in paragraphs_to_be_translated_list:
        #     word_count = len(paragraph.split())
        #     sample_text = """麦卡锡分别与信息论之父克劳德·香农和电气工程先驱纳撒尼尔·罗切斯特合作。麦卡锡在达特茅斯时说服明斯基、香农和罗切斯特帮助他组织"一项为期2个月，10人的人工智能研究，计划在1956年夏天进行。"人工智能这个术语是麦卡锡的创造；他想要将这个领域和一个名为控制论的相关努力区分开来。麦卡锡后来承认这个名字实际上并不受人欢迎——毕竟，目标是真正的，而非"人工"的智能——但"我必须给它一个名字，所以我给它起名为'人工智能'"。"""
        #     translated_paragraph = sample_text[:word_count * 2]  # 确保长度与原文单词数相近
        #     paragraphs_translation_list.append(translated_paragraph)
        # print(paragraphs_translation_list)
        # print('翻译完成，使用了OpenAI！')

    paragraphs_to_be_translated_dict = {}
    for original_paragraph, translated_paragraph, bbox, page_num in zip(
        paragraphs_to_be_translated_list, 
        paragraphs_translation_list, 
        paragraphs_to_be_translated_bbox_list, 
        paragraphs_to_be_translated_page_num_list
    ):
        if page_num not in paragraphs_to_be_translated_dict:
            paragraphs_to_be_translated_dict[page_num] = []
        paragraphs_to_be_translated_dict[page_num].append({
            'original_paragraph': original_paragraph,  # 保存原始文本
            'translated_paragraph': translated_paragraph,
            'bbox': bbox
        })

    # re-open another copy of the pdf
    doc_translated = fitz.open(pdf_path)
    # font_files = [
    #     "Deng.ttf", "Dengb.ttf", "Dengl.ttf", "FZSTK.TTF", "FZYTK.TTF", 
    #     "HYZhongHeiTi-197.ttf", "msyh.ttc", "msyhbd.ttc", "msyhl.ttc",
    #     "simfang.ttf", "simhei.ttf", "simkai.ttf", "SIMLI.TTF", "simsun.ttc",
    #     "SIMYOU.TTF", "SourceHanSansCN-Bold.otf", "SourceHanSansCN-ExtraLight.otf",
    #     "SourceHanSansCN-Heavy.otf", "SourceHanSansCN-Light.otf", 
    #     "SourceHanSansCN-Medium.otf", "SourceHanSansCN-Normal.otf",
    #     "SourceHanSansCN-Regular.otf", "STCAIYUN.TTF", "STFANGSO.TTF",
    #     "STHUPO.TTF", "STKAITI.TTF", "STLITI.TTF", "STSONG.TTF", "STXIHEI.TTF",
    #     "STXINGKA.TTF", "STXINWEI.TTF", "STZHONGS.TTF"
    # ]
    # font_files = ["STFANGSO.TTF"]
    # font_files = ["STXIHEI.TTF"]
    # font_files = ["SourceHanSansCN-Normal.otf"]
    # font_files = ["simsun.ttc"]
    font_files = ["方正宋三_GBK.TTF"]
    
    # font_files = [os.path.join("./fonts", font) for font in font_files]
    original_cwd = os.getcwd()
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_file_dir)
    font_files = [os.path.normpath(os.path.join("./fonts", font)).replace("\\", "/") for font in font_files]
    css = " ".join([f"""@font-face {{font-family: fam{i}; src: url("{font}");}}""" for i, font in enumerate(font_files)])
    
    from tqdm import tqdm

    total_paragraphs = sum(len(paragraphs) for paragraphs in paragraphs_to_be_translated_dict.values())
    pbar = tqdm(total=total_paragraphs, desc="替换页面和段落")
    for i, page in enumerate(doc_translated):
        # 保存原始链接
        original_links = [link for link in page.get_links()]
        # print(f"页面 {i+1} 开始时的链接数: {len(original_links)}")

        if i+1 in paragraphs_to_be_translated_dict:
            for paragraph_data in paragraphs_to_be_translated_dict[i+1]:
                original_paragraph = paragraph_data['original_paragraph']
                translated_paragraph = paragraph_data['translated_paragraph']

                if not is_regular_content(original_paragraph):
                    pbar.update(1)
                    continue

                rect = fitz.Rect(paragraph_data['bbox'])
                page.add_redact_annot(rect, text="")
                page.apply_redactions(images=0)
                
                random_font_index = random.randint(0, len(font_files)-1)
                random_font = f"fam{random_font_index}"
                html = f"""<span style="font-family:{random_font}; line-height: 1.5; letter-spacing: 0.05em;">{translated_paragraph}</span>"""
                
                if rect.height > 30:
                    rect.y1 -= 10
                
                page.insert_htmlbox(rect, html, css=css)
                pbar.update(1)

        # 恢复原始链接
        def restore_links(page, original_links):
            for link in original_links:
                try:
                    page.insert_link(link)
                except Exception as e:
                    print(f"无法恢复链接: {e}")

        try:
            restore_links(page, original_links)
        except Exception as e:
            print(f"无法恢复链接: {e}")

        # 检查页面结束时的链接
        # end_links = page.get_links()
        # print(f"页面 {i+1} 结束时的链接数: {len(end_links)}")

    pbar.close()
    
    # 断言原始文档和翻译后的文档页数相同
    assert len(doc_original) == len(doc_translated)
    print(f"文档页数验证通过：原始文档 {len(doc_original)} 页，翻译后文档 {len(doc_translated)} 页")
                
                
    os.chdir(original_cwd)
    
    
    def create_interleaved_docs(doc_original, doc_translated):
        doc_interleaved = fitz.open()
        doc_interleaved_cn_first = fitz.open()
        
        num_pages = len(doc_original)  # 或 len(doc_translated)，因为它们相同
        
        for i in range(num_pages):
            
            doc_interleaved.insert_pdf(doc_original, from_page=i, to_page=i)
            try:
                # 插入翻译后的页面
                doc_interleaved.insert_pdf(doc_translated, from_page=i, to_page=i)
            except IndexError:
                print(f"警告：处理第 {i+1} 页时出现索引错误，用原始文档页面代替。")
                doc_interleaved.delete_page(-1)
                doc_interleaved.insert_pdf(doc_original, from_page=i, to_page=i)
                
            try:                    
                doc_interleaved_cn_first.insert_pdf(doc_translated, from_page=i, to_page=i)
            except IndexError:
                print(f"警告：处理第 {i+1} 页时出现索引错误，用原始文档页面代替。")
                doc_interleaved_cn_first.delete_page(-1)
                doc_interleaved_cn_first.insert_pdf(doc_original, from_page=i, to_page=i)                    
            doc_interleaved_cn_first.insert_pdf(doc_original, from_page=i, to_page=i)
        
        return doc_interleaved, doc_interleaved_cn_first
    
    def create_concat_docs(doc_original, doc_translated):
        doc_concat = fitz.open()
        doc_concat_cn_first = fitz.open()
        
        doc_concat.insert_pdf(doc_original)
        
        for page in doc_translated:
            try:    
                doc_concat.insert_pdf(doc_translated, from_page=page.number, to_page=page.number)
            except IndexError:
                print(f"警告：处理第 {page.number+1} 页时出现索引错误，用原始文档页面代替。")
                doc_concat.delete_page(-1)
                doc_concat.insert_pdf(doc_original, from_page=page.number, to_page=page.number)
        
        for page in doc_translated:
            try:
                doc_concat_cn_first.insert_pdf(doc_translated, from_page=page.number, to_page=page.number)
            except IndexError:
                print(f"警告：处理第 {page.number+1} 页时出现索引错误，用原始文档页面代替。")
                doc_concat_cn_first.delete_page(-1)
                doc_concat_cn_first.insert_pdf(doc_original, from_page=page.number, to_page=page.number)
        
        doc_concat_cn_first.insert_pdf(doc_original)
        
        return doc_concat, doc_concat_cn_first
    
    
    def add_navigation_among_same_lang(doc):
        num_pages = len(doc)
        next_page_icon_path = os.path.abspath(r"D:\My_Codes\PDF-translation-in-situ\static\next_page-0.125-transparent.png")
        prev_page_icon_path = os.path.abspath(r"D:\My_Codes\PDF-translation-in-situ\static\prev_page-0.125-transparent.png")
        
        for i in range(num_pages):
            page = doc[i]
            page_rect = page.rect
            prev_rect = fitz.Rect(0, page_rect.height / 2, 40, page_rect.height)
            next_rect = fitz.Rect(page_rect.width - 40, page_rect.height / 2, page_rect.width, page_rect.height)
            
            shape = page.new_shape()
            if i > 1:
                page.insert_link({'kind': fitz.LINK_GOTO, 'from': prev_rect, 'page': i - 2})
                shape.draw_rect(prev_rect)
                shape.finish(fill=(0, 0, 1), fill_opacity=0.015, color=(0, 0, 1), stroke_opacity=0.1, dashes="[10 10] 0")
                
                # 添加上一页图标
                icon_size = min(prev_rect.width, prev_rect.height) * 0.6
                icon_rect = fitz.Rect(
                    prev_rect.x0 + (prev_rect.width - icon_size) / 2,
                    prev_rect.y0 + (prev_rect.height - icon_size) / 2,
                    prev_rect.x0 + (prev_rect.width + icon_size) / 2,
                    prev_rect.y0 + (prev_rect.height + icon_size) / 2
                )
                page.insert_image(icon_rect, filename=prev_page_icon_path)
            
            if i < num_pages - 2:
                page.insert_link({'kind': fitz.LINK_GOTO, 'from': next_rect, 'page': i + 2})
                shape.draw_rect(next_rect)
                shape.finish(fill=(0, 0, 1), fill_opacity=0.015, color=(0, 0, 1), stroke_opacity=0.1, dashes="[10 10] 0")
                
                # 添加下一页图标
                icon_size = min(next_rect.width, next_rect.height) * 0.6
                icon_rect = fitz.Rect(
                    next_rect.x0 + (next_rect.width - icon_size) / 2,
                    next_rect.y0 + (next_rect.height - icon_size) / 2,
                    next_rect.x0 + (next_rect.width + icon_size) / 2,
                    next_rect.y0 + (next_rect.height + icon_size) / 2
                )
                page.insert_image(icon_rect, filename=next_page_icon_path)
            
            shape.commit()

    def add_en_cn_pair_back_and_forth(doc, is_interleaved=True):
        num_pages = len(doc)
        icon_path = os.path.abspath(r"D:\My_Codes\PDF-translation-in-situ\static\en-left-64-0.125-transparent.png")
        for i in range(num_pages):
            page = doc[i]
            page_rect = page.rect
            left_rect = fitz.Rect(0, 0, 40, page_rect.height / 2)
            right_rect = fitz.Rect(page_rect.width - 40, 0, page_rect.width, page_rect.height / 2)
            
            if is_interleaved:
                target_page = i + 1 if i % 2 == 0 else i - 1
            else:
                half_pages = num_pages // 2
                target_page = i + half_pages if i < half_pages else i - half_pages
            
            # 循环处理左右矩形
            for rect in [left_rect, right_rect]:
                page.insert_link({'kind': fitz.LINK_GOTO, 'from': rect, 'page': target_page})
                shape = page.new_shape()
                shape.draw_rect(rect)
                shape.finish(fill=(0, 1, 0), fill_opacity=0.015, color=(0, 1, 0), stroke_opacity=0.1, dashes="[10 10] 0")
                shape.commit()
                
                # 添加图标
                icon_size = min(rect.width, rect.height) * 0.6
                icon_rect = fitz.Rect(
                    rect.x0 + (rect.width - icon_size) / 2,
                    rect.y0 + (rect.height - icon_size) / 2,
                    rect.x0 + (rect.width + icon_size) / 2,
                    rect.y0 + (rect.height + icon_size) / 2
                )
                page.insert_image(icon_rect, filename=icon_path)

    def add_navigation(doc):
        num_pages = len(doc)
        next_page_icon_path = os.path.abspath(r"D:\My_Codes\PDF-translation-in-situ\static\next_page-0.125-transparent.png")
        prev_page_icon_path = os.path.abspath(r"D:\My_Codes\PDF-translation-in-situ\static\prev_page-0.125-transparent.png")
        
        for i in range(num_pages):
            page = doc[i]
            page_rect = page.rect
            prev_rect = fitz.Rect(0, page_rect.height / 2, 40, page_rect.height)
            next_rect = fitz.Rect(page_rect.width - 40, page_rect.height / 2, page_rect.width, page_rect.height)
            
            shape = page.new_shape()
            if i > 0:
                page.insert_link({'kind': fitz.LINK_GOTO, 'from': prev_rect, 'page': i - 1})
                shape.draw_rect(prev_rect)
                shape.finish(fill=(0, 0, 1), fill_opacity=0.015, color=(0, 0, 1), stroke_opacity=0.1, dashes="[10 10] 0")
                
                # 添加上一页图标
                icon_size = min(prev_rect.width, prev_rect.height) * 0.6
                icon_rect = fitz.Rect(
                    prev_rect.x0 + (prev_rect.width - icon_size) / 2,
                    prev_rect.y0 + (prev_rect.height - icon_size) / 2,
                    prev_rect.x0 + (prev_rect.width + icon_size) / 2,
                    prev_rect.y0 + (prev_rect.height + icon_size) / 2
                )
                page.insert_image(icon_rect, filename=prev_page_icon_path)
            
            if i < num_pages - 1:
                page.insert_link({'kind': fitz.LINK_GOTO, 'from': next_rect, 'page': i + 1})
                shape.draw_rect(next_rect)
                shape.finish(fill=(0, 0, 1), fill_opacity=0.015, color=(0, 0, 1), stroke_opacity=0.1, dashes="[10 10] 0")
                
                # 添加下一页图标
                icon_size = min(next_rect.width, next_rect.height) * 0.6
                icon_rect = fitz.Rect(
                    next_rect.x0 + (next_rect.width - icon_size) / 2,
                    next_rect.y0 + (next_rect.height - icon_size) / 2,
                    next_rect.x0 + (next_rect.width + icon_size) / 2,
                    next_rect.y0 + (next_rect.height + icon_size) / 2
                )
                page.insert_image(icon_rect, filename=next_page_icon_path)
            
            shape.commit()

    # 主程序中使用
    doc_interleaved, doc_interleaved_cn_first = create_interleaved_docs(doc_original, doc_translated)
    doc_concat, doc_concat_cn_first = create_concat_docs(doc_original, doc_translated)

    # # 检查文档的完整性和完整性
    # def check_doc_integrity(doc, doc_name):
    #     """
    #     检查文档的完整性和完整性
        
    #     Args:
    #     doc (fitz.Document): 要检查的文档
    #     doc_name (str): 文档的名称，用于打印信息
        
    #     Returns:
    #     None
    #     """
    #     print(f"正在检查 {doc_name} 的完整性...")
        
    #     # 检查页数
    #     num_pages = len(doc)
    #     print(f"{doc_name} 的页数: {num_pages}")
        
    #     # 检查元数据
    #     metadata = doc.metadata
    #     print(f"{doc_name} 的元数据:")
    #     for key, value in metadata.items():
    #         print(f"  {key}: {value}")
        
    #     # 检查目录
    #     toc = doc.get_toc()
    #     print(f"{doc_name} 的目录项数: {len(toc)}")
        
    #     # 检查链接
    #     total_links = 0
    #     for page in doc:
    #         links = page.get_links()
    #         total_links += len(links)
    #     print(f"{doc_name} 的总链接数: {total_links}")
        
    #     print(f"{doc_name} 的完整性检查完成\n")

    # # 检查所有文档
    # check_doc_integrity(doc_interleaved, "交错文档")
    # check_doc_integrity(doc_interleaved_cn_first, "中文在前的交错文档")
    # check_doc_integrity(doc_concat, "连续拼接文档")
    # check_doc_integrity(doc_concat_cn_first, "中文在前的连续拼接文档")
    # check_doc_integrity(doc_translated, "翻译文档")
    
    
    add_en_cn_pair_back_and_forth(doc_interleaved, is_interleaved=True)
    add_en_cn_pair_back_and_forth(doc_interleaved_cn_first, is_interleaved=True)
    add_en_cn_pair_back_and_forth(doc_concat, is_interleaved=False)
    add_en_cn_pair_back_and_forth(doc_concat_cn_first, is_interleaved=False)

    add_navigation_among_same_lang(doc_interleaved)
    add_navigation_among_same_lang(doc_interleaved_cn_first)
    
    # 为连续文档添加上一页和下一页的导航
    add_navigation(doc_concat)
    add_navigation(doc_concat_cn_first)
    add_navigation(doc_translated)
    
    original_toc_ = doc_original.get_toc()
    print(original_toc_)
    
    # 修复原始目录中的负页码
    original_toc = []
    
    for item in original_toc_:
        level, title, page = item
        original_toc.append([level, title, max(1, page)])  # 确保页码至少为1
    
    translated_toc = original_toc.copy()  # 使用修复后的目录
        
    # 根据原始文档是否有目录来决定是否设置目录
    if original_toc:
        # 设置交错文档的目录
        interleaved_toc = []
        for item in original_toc:
            level, title, page = item
            interleaved_toc.append([level, title, (page - 1) * 2 + 1])
            interleaved_toc.append([level, title, (page - 1) * 2 + 2])
        doc_interleaved.set_toc(interleaved_toc)

        # 设置中文在前的交错文档的目录
        interleaved_cn_first_toc = []
        for item in original_toc:
            level, title, page = item
            interleaved_cn_first_toc.append([level, title, (page - 1) * 2 + 1])
            interleaved_cn_first_toc.append([level, title, (page - 1) * 2 + 2])
        doc_interleaved_cn_first.set_toc(interleaved_cn_first_toc)

        num_pages = len(doc_original)
        
        # 设置连续拼接文档的目录
        concat_toc = original_toc + [
            [level, title, page + num_pages] 
            for level, title, page in translated_toc
        ]
        doc_concat.set_toc(concat_toc)

        # 设置中文在前的连续拼接文档的目录
        concat_cn_first_toc = translated_toc + [
            [level, title, page + num_pages] 
            for level, title, page in original_toc
        ]
        doc_concat_cn_first.set_toc(concat_cn_first_toc)
    else:
        # 如果原始文档没有目录，则不设置目录
        doc_interleaved.set_toc([])
        doc_interleaved_cn_first.set_toc([])
        doc_concat.set_toc([])
        doc_concat_cn_first.set_toc([])

    # 在函数返回值中添加 doc_concat 和 doc_concat_cn_first
    # return paragraphs_, page_nums, toc_titles, doc, all_blocks_aspect_ratio, doc_translated, doc_interleaved, doc_concat, doc_concat_cn_first
    return paragraphs_, page_nums, toc_titles, doc, all_blocks_aspect_ratio, doc_translated, doc_interleaved, doc_interleaved_cn_first, doc_concat, doc_concat_cn_first



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract paragraphs from PDF.')
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument('pdf_path', help='the path of the file to process')
    # set default pdf path
    parser.add_argument('--pdf_path', type=str, help='path to pdf file')
    # output as txt flag with default False
    parser.add_argument('--txt', action='store_true', help='output as txt file')
    # output as csv flag with default False
    parser.add_argument('--csv', action='store_true', help='output as csv file')
    # output as html flag with default False
    parser.add_argument('--html', action='store_true', help='output as html file')
    # output layout with marks flag with default False
    parser.add_argument('--layout', action='store_true', help='output layout with marks')
    # plot the paragraph length distribution
    parser.add_argument('--plot', action='store_true', help='plot the paragraph length distribution')
    # plot aspect_ratio distribution
    parser.add_argument('--aspect_ratio', action='store_true', help='plot the aspect_ratio distribution')
    # only save txt, nothing else
    parser.add_argument('--txt_only', action='store_true', help='only save txt, nothing else')

    parser.add_argument('--translation_only', action='store_true', help='only translation, nothing else')
    parser.add_argument('--heuristic', action='store_true', help='use heuristic to split paragraphs')
    parser.add_argument('--dump_json_only', action='store_true', help='只生成和保存发送到服务器的数据的JSON副本')
    parser.add_argument('--use_openai', action='store_true', help='使用OpenAI进行翻译')
    parser.add_argument('--en_cn_concat_only', action='store_true', help='只输出英文PDF后整体追加中文翻译')

    args = parser.parse_args()

    args.translation_only = True  # 仅进行翻译
    
    dump_json_only = args.dump_json_only
    use_openai = args.use_openai
    en_cn_concat_only = args.en_cn_concat_only
    
    # 定义全局变量
    global existing_data
    existing_data = {}
    requests_dump_file = 'requests_dump.json'

    if dump_json_only:
        # 保存发送到服务器的数据的JSON副本
        if os.path.exists(requests_dump_file):
            with open(requests_dump_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                
    if not use_openai and not dump_json_only:     # 启动翻译服务
        
        # 检查翻译服务是否已经在运行
        script_name = "server_batching.py"
        if is_script_running(script_name):
            print("翻译服务已在运行")
        else:
            print("正在启动翻译服务")
            subprocess.Popen(["python", "D:\\My_Codes\\Translation_Service\\server_batching.py"])
        
    
    output_plot = args.plot
    output_layout = args.layout
    output_txt = args.txt
    output_csv = args.csv
    output_html = args.html
    output_aspect_ratio = args.aspect_ratio
    heuristic = args.heuristic


    if                             111111111111111111111                                :
        output_plot = 1
        output_layout = 1
        output_txt = 1
        output_csv = 1
        output_html = 1
        output_aspect_ratio = 1

    if args.txt_only:
        output_plot = 0
        output_layout = 0
        output_csv = 0
        output_html = 0
        output_aspect_ratio = 0
    
    if heuristic:
        output_plot = 0
        # output_html = 0
        output_aspect_ratio = 0


    if not output_txt and not output_csv and not output_html and not output_layout and not output_plot and not output_aspect_ratio:
        print("Please specify at least one output format.")
        sys.exit(0)


    
    # if is_scanned_pdf(pdf_path):
    #     # write the pdf name to a file
    #     with open('__scanned_pdf.txt', 'a') as f:
    #         f.write(pdf_path + '\n\n')
    #     print("The provided PDF appears to be a scan and does not contain extractable text.")
    #     time.sleep(3)
    #     sys.exit(0)

    

    pdf_path = args.pdf_path
    print(pdf_path)
    
    def process_pdf(pdf_path):
        # 检查当前PDF路径是否在跳过列表中或以跳过路径结尾
        normalized_pdf_path = os.path.normpath(pdf_path)
        if skip_paths and any(normalized_pdf_path.endswith(os.path.normpath(skip_path)) for skip_path in skip_paths):
            print(f"\033[93m跳过处理 PDF in skip_path_list.txt: {normalized_pdf_path}\033[0m")
            # 记录被跳过的路径
            with open('_skipped_pdfs.log', 'a', encoding='utf-8') as log:
                log.write(f"{normalized_pdf_path}\n")
            return

        # 检查当前PDF路径是否在包含列表中
        if include_paths and not any(normalized_pdf_path.endswith(os.path.normpath(include_path)) for include_path in include_paths):
            print(f"\033[93m跳过处理 PDF 不在 include_path_list.txt 中: {normalized_pdf_path}\033[0m")
            # 记录被跳过的路径
            with open('_skipped_pdfs_not_included.log', 'a', encoding='utf-8') as log:
                log.write(f"{normalized_pdf_path}\n")
            return

        if is_scanned_pdf(pdf_path):
            print(f"\033[93m扫描PDF，无法提取文本: {pdf_path}\033[0m")
            with open('_scanned_pdfs_case.log', 'a', encoding='utf-8') as log:
                log.write(f"{pdf_path}\n")
            return
        
        print(f"\033[92m处理PDF: {pdf_path}\033[0m")
        if dump_json_only:
            try:
                extract_paragraphs(pdf_path, heuristic)
                print(f"\033[92m成功提取段落并 dump requests: {pdf_path}\033[0m")
                with open('_processed_pdfs_requests_dump.log', 'a', encoding='utf-8') as log:
                    log.write(f"{pdf_path}\n")
            except Exception as e:
                print(f"\033[93m提取段落时出错: {str(e)}\033[0m")
                with open('_error_pdfs_requests_dump.log', 'a', encoding='utf-8') as log:
                    log.write(f"{pdf_path}: {str(e)}\n")
            return
        
        try:
            paragraphs, page_nums, toc_titles, doc, all_blocks_aspect_ratio,\
            doc_translated, doc_interleaved, doc_interleaved_cn_first, doc_concat, doc_concat_cn_first = extract_paragraphs(pdf_path, heuristic)
            
            output_dir = os.path.dirname(pdf_path)
            base_name = os.path.basename(pdf_path)[:-4]
            
            if en_cn_concat_only:
                doc_concat.subset_fonts(verbose=True, fallback=True)
                doc_concat.save(os.path.join(output_dir, f"{base_name}_en+zh-concat.pdf"), garbage=4, deflate=True)
            else:
                doc_translated.subset_fonts(verbose=True, fallback=True)
                doc_translated.save(os.path.join(output_dir, f"{base_name}_zh_trans.pdf"), garbage=4, deflate=True)

                doc_concat.subset_fonts(verbose=True, fallback=True)
                doc_concat.save(os.path.join(output_dir, f"{base_name}_en+zh-concat.pdf"), garbage=4, deflate=True)

                doc_concat_cn_first.subset_fonts(verbose=True, fallback=True)
                doc_concat_cn_first.save(os.path.join(output_dir, f"{base_name}_zh+en-concat.pdf"), garbage=4, deflate=True)

                doc_interleaved.subset_fonts(verbose=True, fallback=True)
                doc_interleaved.save(os.path.join(output_dir, f"{base_name}_en+zh-interleaved.pdf"), garbage=4, deflate=True)
                
                doc_interleaved_cn_first.subset_fonts(verbose=True, fallback=True)
                doc_interleaved_cn_first.save(os.path.join(output_dir, f"{base_name}_zh+en-interleaved.pdf"), garbage=4, deflate=True)
            
            print(f"\033[92m成功处理PDF: {pdf_path}\033[0m")
            with open('_processed_pdfs.log', 'a', encoding='utf-8') as log:
                log.write(f"{pdf_path}\n")
                
        except Exception as e:
            print(f"\033[93m处理PDF时出错: {pdf_path}\033[0m")
            error_message = traceback.format_exc()
            print(f"\033[93m错误信息:\n{error_message}\033[0m")
            with open('_error_pdfs.log', 'a', encoding='utf-8') as log:
                log.write(f"处理PDF时出错: {pdf_path}\n")
                log.write(f"错误信息:\n{error_message}\n")
                log.write("-" * 50 + "\n")  # 添加分隔线以区分不同的错误

    if not pdf_path == None and os.path.exists(pdf_path):
        process_pdf(pdf_path)
    else:
        pdf_paths = pyperclip.paste().splitlines()
        print('pdf_paths from clipboard: ', pdf_paths)
        
        start_time = time.time()
        
        pbar = tqdm(pdf_paths, desc="PDF文件数进度", colour="green")
        for clip_path in pbar:
            clip_path = clip_path.strip('"')
            if os.path.isdir(clip_path):
                for root, dirs, files in os.walk(clip_path):
                    for file in files:
                        if file.lower().endswith('.pdf'):
                            pdf_path = os.path.join(root, file)
                            if not os.path.exists(pdf_path):
                                print(f"\033[92m无效的PDF路径: {pdf_path}\033[0m")
                                with open('_invalid_pdfs_case.log', 'a', encoding='utf-8') as log:
                                    log.write(f"{pdf_path}\n")
                                continue
                            process_pdf(pdf_path)
            elif os.path.isfile(clip_path) and clip_path.lower().endswith('.pdf'):
                if not os.path.exists(clip_path):
                    print(f"\033[92m无效的PDF路径: {clip_path}\033[0m")
                    with open('_invalid_pdfs_case.log', 'a', encoding='utf-8') as log:
                        log.write(f"{clip_path}\n")
                    continue
                process_pdf(clip_path)
            else:
                print(f"\033[92m无效的路径: {clip_path}\033[0m")
                with open('_invalid_paths_case.log', 'a', encoding='utf-8') as log:
                    log.write(f"{clip_path}\n")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_time_hms = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
        print(f"\n总耗时: {elapsed_time:.2f} 秒 ({elapsed_time_hms})")
        
        with open('_task_duration.log', 'w', encoding='utf-8') as log:
            log.write(f"总耗时: {elapsed_time:.2f} 秒 ({elapsed_time_hms})\n")

    
    if dump_json_only:
        with open(requests_dump_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)
        sys.exit(0)
        
        
    if args.translation_only:
        sys.exit(0)


    def remove_extra_spaces(text):
        if not has_chinese(text):
            words = [w for w in text.split(' ') if w.strip()]
            return ' '.join(words)
        else:
            return text

    paragraphs = [remove_extra_spaces(p) for p in paragraphs]

    # for i, paragraph in enumerate(paragraphs):
    #     # print(f"Paragraph {i+1}: {paragraph[:100]}")
    #     # print()
    #     pass

    if output_aspect_ratio:
        # plot the the distribution of aspect ratio of blocks
        plt.figure()
        plt.hist(all_blocks_aspect_ratio, bins=10)
        plt.xlabel('Aspect Ratio')
        plt.ylabel('Number of blocks')
        plt.title('Aspect Ratio Distribution')
        plt.savefig('_aspect_ratio_distribution_' + pdf_name[:-4] + '.png')
        # save the proportion of blocks with aspect ratio > 20
        blocks_aspect_ratio_above_20 = [aspect_ratio for aspect_ratio in all_blocks_aspect_ratio if aspect_ratio > 20]
        proportion_blocks_aspect_ratio_above_20 = len(blocks_aspect_ratio_above_20)/len(all_blocks_aspect_ratio)
        if proportion_blocks_aspect_ratio_above_20 > 0.5:
            with open('_with_high_proportion_blocks_aspect_ratio_above_20_.txt', 'a') as f:
                f.write(str(round(proportion_blocks_aspect_ratio_above_20, 2))+ ' ' + pdf_name[:-4] + '\n\n')
        else:
            with open('_with_low_proportion_blocks_aspect_ratio_above_20_.txt', 'a') as f:
                f.write(str(round(proportion_blocks_aspect_ratio_above_20, 2))+ ' ' + pdf_name[:-4] + '\n\n')

    
    # Get the lengths of paragraphs
    # check if chinese
    # para_lengths = np.array([len(p.split()) for p in paragraphs])
    para_lengths = np.array([])
    for p in paragraphs:
        if has_chinese(p):
            para_lengths = np.append(para_lengths, len(re.findall(r'[\u4e00-\u9fff]+', p)))
        else:
            para_lengths = np.append(para_lengths, len(p.split()))
    # Create an array of cutoff values to consider
    cutoffs = np.arange(1, para_lengths.max()+1)
    # For each cutoff, count the number of paragraphs with a word count greater than the cutoff
    counts = [np.sum(para_lengths > cutoff) for cutoff in cutoffs]

    # Find the elbow point
    knee_locator = KneeLocator(cutoffs, counts, curve='convex', direction='decreasing')
    elbow = knee_locator.knee
    elbow_original = elbow
    print(f'original Elbow point is: {elbow}')
    # elbow should > 15 but < 40
    if elbow < 15:
        elbow = 15
    elif elbow > 35:
        elbow = 35
    print(f'corrected Elbow point is: {elbow}')

    if output_plot:
        # plot the paragraph length distribution
        plt.figure()
        plt.hist([len(p.split()) for p in paragraphs], bins=10)
        plt.xlabel('Number of words')
        plt.ylabel('Number of paragraphs')
        plt.title('Paragraph length distribution')
        plt.savefig('_paragraph_length_distribution_' + pdf_name[:-4] + '.png')

        # Filter paragraphs that are below the elbow cutoff
        filtered_paragraphs = [p for p in paragraphs if len(p.split()) <= elbow]
        # Sort the filtered paragraphs by their word count in descending order
        filtered_paragraphs.sort(key=lambda p: len(p.split()), reverse=True)
        # Save the filtered paragraphs to a text file, with each line starting with the word count
        with open(f'_filtered_paragraphs_' + pdf_name[:-4] + '_elbow_original_{elbow_original}.txt', 'w', encoding='utf-8') as f:
            for p in filtered_paragraphs:
                f.write(str(len(p.split())) + ' ' + p + '\n\n')

        # Plot the relation
        plt.figure()
        plt.plot(cutoffs, counts)
        plt.xlabel('Word Number Cutoff')
        plt.ylabel('Number of Paragraphs above Cutoff')
        plt.title('Relation between Word Number Cutoff and Paragraph Count')

        # Mark the elbow point on the plot
        plt.plot(elbow, knee_locator.knee_y, 'ro')  # mark the point with a red circle
        plt.annotate('Elbow (cutoff: %d)' % elbow,  # text to display
                    (elbow, knee_locator.knee_y),  # point to mark
                    textcoords="offset points",  # how to position the text
                    xytext=(-10,10),  # distance from text to points (x,y)
                    ha='center')  # horizontal alignment can be left, right or center

        plt.savefig('_cutoff_vs_paragraph_count_' + pdf_name[:-4] + '.png')

    if output_layout:
        # base_name = pdf_name.split('/')[-1]
        # doc.save("_plain_layout_marked_" + pdf_name[:-4] + ".pdf")
        # reopen it
        # doc = fitz.open("_plain_layout_marked_" + pdf_name[:-4] + ".pdf")
        # doc.subset_fonts()  # build subset fonts to reduce file size
        # pdf = fitz._as_pdf_document(doc)  # access underlying PDF document of the general Document
        # fitz.mupdf.pdf_subset_fonts2(pdf, list(range(doc.page_count)))  # create font subsets   
        # doc.ez_save("_plain_layout_marked_" + pdf_name[:-4] + ".pdf")
        doc.save("_plain_layout_marked_" + pdf_name[:-4] + ".pdf", garbage=4, deflate=True)

    if output_txt:
        # write to text file
        with open(pdf_name[:-4] + '.txt', 'w', encoding='utf-8') as f:
            f.write("\n\n".join(paragraphs))
    
    if output_csv:
        # write to csv file
        with open(pdf_name[:-4] + '.csv', 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["page_num", "toc_title", "paragraph", "paragraph_translation", "embedding", "embedding_translation", "num_tokens", "num_tokens_translation", "above_elbow", "file_name"])
            for i in range(len(paragraphs)):
                if not paragraphs[i]:
                    continue
                paragraph_translation = ""
                above_elbow = len(paragraphs[i].split()) > elbow
                if not has_chinese(paragraphs[i]):
                    if not above_elbow:
                        paragraph_translation = paragraphs[i]
                writer.writerow([page_nums[i], toc_titles[i], paragraphs[i], paragraph_translation, "", "", "", "", above_elbow, pdf_name])
    
    if output_html:
        # create html string with paragraphs wrapped in <p> tags
        html_string = '''<html><head>
            <style>
            body {
                margin: 50px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
            }
            div.paragraph-container {
                margin-bottom: 20px;
                max-width: 800px;
            }
            p.paragraph {
                text-align: justify;
                font-size: 18px;
                line-height: 1.6;
            }
            </style>
            </head><body><div class='paragraph-container'>'''
        for p in paragraphs:
            html_string += f"<p class='paragraph'>{p}</p>"
        html_string += "</div></body></html>"

        # write html to file
        with open(pdf_name[:-4] + '.html', 'w', encoding='utf-8') as f:
            f.write(html_string)

        # copy to clipboard
        # if copy:
        #     pyperclip.copy(html_string)
        #     print('Copied to clipboard')
        # else:
    




# import asyncio

# def translate_paragraphs_openai(paragraphs):
#     client = OpenAI(
#         api_key="sk-.....",
#         base_url='https://sapi.onechats.top/v1/'
#     )
    
#     async def translate_paragraph(para, index):
#         if is_regular_content(para):
#             prompt = f"将以下文本翻译成中文：\n\n{para}\n\n翻译："
            
#             chat_completion = await client.chat.completions.create(
#                 messages=[
#                     {
#                         "role": "user",
#                         "content": prompt,
#                     }
#                 ],
#                 model="gpt-3.5-turbo",
#             )
            
#             translated_para = chat_completion.choices[0].message.content.strip()
#             print(f"已翻译段落 {index+1}/{len(paragraphs)}:\n{para}\n{translated_para}")
#             return translated_para
#         else:
#             return para

#     async def translate_all():
#         tasks = [translate_paragraph(para, i) for i, para in enumerate(paragraphs)]
#         return await asyncio.gather(*tasks[:100])  # 最多并行100个任务

#     translated_paragraphs = asyncio.run(translate_all())
    
#     return translated_paragraphs