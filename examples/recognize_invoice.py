#!/usr/bin/env python3
"""
MinerU VLM 发票识别示例
使用 Xinference 的 MinerU VLM 模型识别发票图片
"""
import base64
from xinference.client import Client


def encode_image_to_base64(image_path):
    """将图片编码为 base64"""
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    return base64.b64encode(image_bytes).decode('utf-8')


def recognize_invoice(image_path, model_uid=None):
    """
    使用 MinerU VLM 识别发票
    
    Args:
        image_path: 发票图片路径
        model_uid: 模型UID，如果为None则自动查找或启动
    
    Returns:
        识别结果文本
    """
    # 连接到 Xinference
    client = Client("http://localhost:9997")
    
    # 如果没有指定 model_uid，尝试获取已启动的 mineru-vlm 模型
    if model_uid is None:
        try:
            models = client.list_models()
            print("已启动的模型:", list(models.keys()))
            
            # list_models() 返回的是 {model_name: model_info} 字典
            if 'mineru-vlm' in models:
                model_uid = models['mineru-vlm']['id']
                print(f"✓ 找到已启动的模型: {model_uid}")
            else:
                print("× 未找到已启动的 mineru-vlm 模型")
                print("请先手动启动模型：")
                print("  xinference launch --model-name mineru-vlm --model-engine vllm --model-type LLM")
                raise ValueError("未找到 mineru-vlm 模型，请先启动模型")
        except Exception as e:
            print(f"错误: {e}")
            raise
    
    # 获取模型
    model = client.get_model(model_uid)
    
    # 编码图片
    print(f"正在读取图片: {image_path}")
    img_base64 = encode_image_to_base64(image_path)
    
    # 构建消息
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "请识别这张发票，提取以下信息：\n1. 发票代码\n2. 发票号码\n3. 开票日期\n4. 购买方名称\n5. 销售方名称\n6. 金额\n7. 税额\n8. 价税合计\n\n请以结构化格式输出。"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}"
                    }
                }
            ]
        }
    ]
    
    # 调用模型
    print("正在识别发票...")
    response = model.chat(
        messages=messages,
        generate_config={
            "max_tokens": 2048,
            "temperature": 0.1,  # 低温度确保结果稳定
        }
    )
    
    # 提取结果
    result = response['choices'][0]['message']['content']
    return result


if __name__ == "__main__":
    # 发票图片路径
    invoice_image = "fapiao.png"
    
    try:
        # 识别发票
        result = recognize_invoice(invoice_image)
        
        # 显示结果
        print("\n" + "="*50)
        print("发票识别结果：")
        print("="*50)
        print(result)
        print("="*50)
        
    except FileNotFoundError:
        print(f"错误: 找不到图片文件 '{invoice_image}'")
        print("请确保 fapiao.png 文件在当前目录下")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
