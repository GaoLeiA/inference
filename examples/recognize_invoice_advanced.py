#!/usr/bin/env python3
"""
MinerU VLM 发票识别示例 - 增强版
支持流式输出、多种提示模板、批量处理
"""
import base64
import argparse
import json
from pathlib import Path
from xinference.client import Client


# 不同的提示模板
PROMPT_TEMPLATES = {
    "basic": "请识别这张发票图片中的所有文字。",
    
    "structured": """请识别这张发票，提取以下信息：
1. 发票代码
2. 发票号码
3. 开票日期
4. 购买方名称
5. 销售方名称
6. 金额
7. 税额
8. 价税合计

请以结构化格式输出。""",
    
    "json": """请识别这张发票，并以JSON格式返回以下信息：
{
    "invoice_code": "发票代码",
    "invoice_number": "发票号码",
    "date": "开票日期",
    "buyer": "购买方名称",
    "seller": "销售方名称",
    "amount": "金额",
    "tax": "税额",
    "total": "价税合计"
}""",
    
    "table": """请识别这张发票，提取所有商品明细，包括：
- 商品名称
- 规格型号
- 单位
- 数量
- 单价
- 金额

并以表格形式返回。"""
}


def encode_image_to_base64(image_path):
    """将图片编码为 base64"""
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    return base64.b64encode(image_bytes).decode('utf-8')


def get_or_launch_model(client, model_name="mineru-vlm"):
    """获取已启动的模型"""
    try:
        models = client.list_models()
        print("已启动的模型:", list(models.keys()))
        
        # list_models() 返回的是 {model_name: model_info} 字典
        if model_name in models:
            model_uid = models[model_name]['id']
            print(f"✓ 找到已启动的模型: {model_uid}")
            return client.get_model(model_uid)
        else:
            print(f"✗ 未找到已启动的 {model_name} 模型")
            print("请先手动启动模型：")
            print(f"  xinference launch --model-name {model_name} --model-engine vllm --model-type LLM")
            raise ValueError(f"未找到 {model_name} 模型，请先启动模型")
    except ValueError:
        raise
    except Exception as e:
        print(f"✗ 获取模型失败: {e}")
        raise


def recognize_invoice(
    image_path,
    prompt_template="structured",
    stream=False,
    max_tokens=2048,
    temperature=0.1,
    client_url="http://localhost:9997"
):
    """
    识别发票
    
    Args:
        image_path: 图片路径
        prompt_template: 提示模板类型 (basic/structured/json/table)
        stream: 是否使用流式输出
        max_tokens: 最大生成token数
        temperature: 温度参数
        client_url: Xinference 服务地址
    """
    # 连接到 Xinference
    print(f"连接到 Xinference: {client_url}")
    client = Client(client_url)
    
    # 获取模型
    model = get_or_launch_model(client)
    
    # 获取提示词
    prompt = PROMPT_TEMPLATES.get(prompt_template, PROMPT_TEMPLATES["structured"])
    
    # 编码图片
    print(f"读取图片: {image_path}")
    img_base64 = encode_image_to_base64(image_path)
    
    # 构建消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                }
            ]
        }
    ]
    
    # 配置
    generate_config = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    
    print(f"识别中... (stream={stream})")
    print("-" * 60)
    
    if stream:
        # 流式输出
        full_response = ""
        for chunk in model.chat(messages=messages, generate_config=generate_config):
            if chunk['choices'][0].get('delta', {}).get('content'):
                content = chunk['choices'][0]['delta']['content']
                print(content, end='', flush=True)
                full_response += content
        print()  # 换行
        return full_response
    else:
        # 非流式输出
        response = model.chat(messages=messages, generate_config=generate_config)
        result = response['choices'][0]['message']['content']
        print(result)
        return result


def main():
    parser = argparse.ArgumentParser(description="使用 MinerU VLM 识别发票")
    parser.add_argument("image", help="发票图片路径")
    parser.add_argument(
        "--template", "-t",
        choices=list(PROMPT_TEMPLATES.keys()),
        default="structured",
        help="提示模板类型"
    )
    parser.add_argument(
        "--stream", "-s",
        action="store_true",
        help="使用流式输出"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="最大生成token数"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="温度参数 (0.0-2.0)"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:9997",
        help="Xinference 服务地址"
    )
    parser.add_argument(
        "--output", "-o",
        help="保存结果到文件"
    )
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not Path(args.image).exists():
        print(f"✗ 错误: 找不到图片文件 '{args.image}'")
        return 1
    
    try:
        # 识别发票
        result = recognize_invoice(
            image_path=args.image,
            prompt_template=args.template,
            stream=args.stream,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            client_url=args.url
        )
        
        # 保存结果
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"\n✓ 结果已保存到: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
