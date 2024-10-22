from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
import time
import json
import os
import hashlib
import requests
import glob
from datetime import datetime
import math
import GPUtil


start_time = time.time()
from transformers import T5ForConditionalGeneration, T5Tokenizer
end_time = time.time()
print(f"导入 transformers 模块耗时: {end_time - start_time:.2f} 秒")

app = FastAPI()

# 模型加载和初始化
print("开始加载模型...")
start_time = time.time()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model_name = 'utrobinmv/t5_translate_en_ru_zh_small_1024'
# batch_size = 16
model_name = 'utrobinmv/t5_translate_en_ru_zh_base_200'
batch_size = 3
# model_name = 'utrobinmv/t5_translate_en_ru_zh_large_1024'
# batch_size = 2

model_load_start = time.time()
# model = T5ForConditionalGeneration.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name, local_files_only=True)
model_load_end = time.time()
print(f"模型加载耗时: {model_load_end - model_load_start:.2f} 秒")

tokenizer_load_start = time.time()
# tokenizer = T5Tokenizer.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name, local_files_only=True)
tokenizer_load_end = time.time()
print(f"分词器加载耗时: {tokenizer_load_end - tokenizer_load_start:.2f} 秒")

model_to_device_start = time.time()
model = model.to(device)
model_to_device_end = time.time()
print(f"模型移至 {device} 耗时: {model_to_device_end - model_to_device_start:.2f} 秒")

end_time = time.time()
print(f"总初始化耗时: {end_time - start_time:.2f} 秒")

class TranslationRequest(BaseModel):
    paragraphs: List[str]
    target_lang: str = "zh"
    test_mode: bool = False  # 添加测试模式标志

# 缓存文件路径
# CACHE_DIR = os.path.dirname(__file__)
CACHE_DIR = "./"
CACHE_PREFIX = "translation_cache_"

# 加载缓存
def load_cache():
    cache = {}
    cache_files = sorted(glob.glob(os.path.join(CACHE_DIR, f"{CACHE_PREFIX}*.json")))
    print("加载的缓存文件:")
    for file in cache_files:
        print(f"- {os.path.basename(file)}")
        with open(file, "r", encoding="utf-8") as f:
            cache.update(json.load(f))
    return cache
# 全局变量用于存储上一次保存的缓存文件名
last_saved_cache_file = None

def save_incremental_cache(new_cache, old_cache):
    global last_saved_cache_file
    incremental_cache = {k: v for k, v in new_cache.items() if k not in old_cache or old_cache[k] != v}
    if incremental_cache:
        # 删除上一次保存的缓存文件
        if last_saved_cache_file and os.path.exists(last_saved_cache_file):
            os.remove(last_saved_cache_file)
            print(f"已删除上一次保存的缓存文件: {last_saved_cache_file}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cache_file = os.path.join(CACHE_DIR, f"{CACHE_PREFIX}_{timestamp}.json")
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(incremental_cache, f, ensure_ascii=False, indent=2)
        print(f"已保存 {len(incremental_cache)} 条新的缓存条目")
        
        for i, (k, v) in enumerate(incremental_cache.items()):
            if i >= 3:
                break
            print(f"{i+1}:{v[:15]}", end="   ")
        print(f"...")

        # 更新上一次保存的缓存文件名
        last_saved_cache_file = cache_file
    else:
        print("没有新的缓存条目需要保存")
        
# 生成缓存键
def generate_cache_key(text, target_lang, model_name):
    text_hash = hashlib.md5(text.encode()).hexdigest()
    return f"{text_hash}_{target_lang}_{model_name}"

# 加载缓存
translation_cache = load_cache()
original_cache = translation_cache.copy()

def split_text(text, max_length=512):
    # Split text into chunks of max_length tokens
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) > max_length:
        print(f"输入文本过长，需要分块处理！")
        print(f"输入文本长度: {len(tokens)}")
        print(f"输入文本: {text}")
    chunks = []
    for i in range(0, len(tokens), max_length):
        chunk = tokenizer.decode(tokens[i:i + max_length], skip_special_tokens=True)
        chunks.append(chunk)
    return chunks

@app.post("/translate")
async def translate(request: TranslationRequest):
    try:
        # 如果是测试模式，直接返回原始段落
        if request.test_mode:
            return {"translated_paragraphs": request.paragraphs}

        prefix = f'translate to {request.target_lang}: '
        translated_paragraphs = []
        total_time = 0
        total_input_words = 0
        total_output_chars = 0
        gpu_load_percentages = []
        gpu_memory_percentages = []
        # output_speeds = []
        
        # 创建进度条
        from tqdm import tqdm
        pbar = tqdm(total=len(request.paragraphs), desc="翻译进度")
        
        # 首先对所有段落进行分块
        all_chunks = []
        chunk_map = {}
        all_chunks_translations = []
        
        for idx, text in enumerate(request.paragraphs):
            chunks = split_text(text)
            all_chunks.extend(chunks)
            chunk_map[idx] = len(chunks)
        
        # 对分块后的文本进行批处理
        for i in range(0, len(all_chunks), batch_size):
            start_time = time.time()
            print(f'正在处理第 {i + 1} / {len(all_chunks)} 批次')
            batch = all_chunks[i:i+batch_size]
            
            # 检查缓存并翻译未缓存的块
            uncached_batch = []
            uncached_indices = []
            batch_translations = [None] * len(batch)
            for idx, chunk in enumerate(batch):
                cache_key = generate_cache_key(chunk, request.target_lang, model_name)
                if cache_key in translation_cache:
                    batch_translations[idx] = translation_cache[cache_key]
                else:
                    uncached_batch.append(chunk)
                    uncached_indices.append(idx)
            
            if uncached_batch:
                input_texts = [prefix + chunk for chunk in uncached_batch]
                inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                
                print("输入和输出文本:")
                with torch.no_grad():
                    # output_ids = model.generate(**inputs, max_new_tokens=1024)
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=1024,  # Increase max_new_tokens
                        num_beams=4,          # Use beam search
                        no_repeat_ngram_size=3,  # Avoid repeating 3-grams
                        length_penalty=0.6,   # Encourage longer outputs
                        early_stopping=False  # Disable early stopping
                    )
                    # output_ids = model.generate(**inputs, max_new_tokens=512*1.3)
                    # output_ids = model.generate(**inputs, max_length=512)

                uncached_translations = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                
                for i, (input_text, output_text) in enumerate(zip(input_texts, uncached_translations), 1):
                    print(f"  {i}. 输入: {input_text[len(prefix):]}\n")
                    print(f"     输出: {output_text}\n")
                
                gpus = GPUtil.getGPUs()
                gpu = gpus[0]
                gpu_load_percentage = gpu.load * 100
                gpu_memory_percentage = (gpu.memoryUsed / gpu.memoryTotal) * 100
                gpu_temperature = gpu.temperature
                gpu_load_percentages.append(gpu_load_percentage)
                gpu_memory_percentages.append(gpu_memory_percentage)
                print(f"GPU利用率: {gpu_load_percentage:.2f}%, 显存使用率: {gpu_memory_percentage:.2f}%, GPU温度: {gpu_temperature:.2f}°C")
                
                for idx, (chunk, translation) in enumerate(zip(uncached_batch, uncached_translations)):
                    cache_key = generate_cache_key(chunk, request.target_lang, model_name)
                    translation_cache[cache_key] = translation
                    batch_translations[uncached_indices[idx]] = translation
                    
            all_chunks_translations.extend(batch_translations)
            
            end_time = time.time()
            batch_time = end_time - start_time
            total_time += batch_time
            
            # 计算输入单词数和输出字符数
            batch_input_words = sum(len(chunk.split()) for chunk in batch)
            batch_output_chars = sum(len(translation) for translation in batch_translations)
            total_input_words += batch_input_words
            total_output_chars += batch_output_chars
            
            avg_time_per_chunk = batch_time / len(batch)
            if uncached_batch:
                input_words_per_sec = batch_input_words / batch_time
                output_chars_per_sec = batch_output_chars / batch_time
                # output_speeds.append(output_chars_per_sec)
            else:
                print(f"\033[93m批处理时间异常短！{batch_time:.4f}秒\033[0m")
                input_words_per_sec = output_chars_per_sec = "infinite"

            print(f'批处理时间: {batch_time:.2f}秒, 每块平均时间: {avg_time_per_chunk:.2f}秒')
            print(f'输入速度: {"无穷大" if input_words_per_sec == "infinite" else int(input_words_per_sec)} 单词/秒, \
                  输出速度: {"无穷大" if output_chars_per_sec == "infinite" else int(output_chars_per_sec)} 字符/秒')
            pbar.update(len(batch))
            print('\n')
        
        # 重新组合翻译后的块
        # print('all_chunks_translations:')
        # print(all_chunks_translations)
        current_chunk = 0
        for idx, num_chunks in chunk_map.items():
            translated_paragraph = " ".join(all_chunks_translations[current_chunk:current_chunk + num_chunks])
            translated_paragraphs.append(translated_paragraph)
            current_chunk += num_chunks
        
        # print('translated_paragraphs:')
        # print(translated_paragraphs)
        # 关闭进度条
        pbar.close()
        
        # if total_time > 1e-9:  # 使用一个很小的阈值而不是0
        #     avg_time_overall = total_time / len(request.paragraphs)
        #     overall_input_words_per_sec = total_input_words / total_time
        #     overall_output_chars_per_sec = total_output_chars / total_time
        # else:
        #     print(f"\033[93m总处理时间异常短！{total_time:.4f}秒\033[0m")
        #     avg_time_overall = 0
        #     overall_input_words_per_sec = "infinite"
        #     overall_output_chars_per_sec = "infinite"
        
        # print(f'总处理时间: {total_time:.2f}秒, 整体每段平均时间: {avg_time_overall:.2f}秒')
        # print(f'整体输入速度: {"无穷大" if overall_input_words_per_sec == "infinite" else int(overall_input_words_per_sec)} 单词/秒, \
        #       整体输出速度: {"无穷大" if overall_output_chars_per_sec == "infinite" else int(overall_output_chars_per_sec)} 字符/秒')
        
        # 保存增量缓存
        save_incremental_cache(translation_cache, original_cache)
        
        return {"translated_paragraphs": translated_paragraphs}
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"发生错误：\n{error_details}")
        raise HTTPException(status_code=500, detail=f"翻译过程中发生错误：{str(e)}\n\n详细信息：{error_details}")

@app.get("/")
async def check_server():
    """
    检查服务器是否准备就绪的路由
    """
    return {"status": "Server is ready"}

if __name__ == "__main__":
    import uvicorn
    print("Server is starting...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    print("Server is ready!")  # This line won't be reached in practice