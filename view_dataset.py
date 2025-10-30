#!/usr/bin/env python3
"""
查看 CoachMe 数据集内容的简单脚本
用法: python view_dataset.py
"""

import pickle
import torch
import json
from pathlib import Path

def view_pkl_dataset(pkl_path, show_details=True, max_samples=3):
    """
    查看 PKL 数据集的内容
    
    Args:
        pkl_path: pkl 文件路径
        show_details: 是否显示详细信息
        max_samples: 最多显示几个样本的详细信息
    """
    print(f"{'='*80}")
    print(f"📦 查看数据集: {pkl_path}")
    print(f"{'='*80}\n")
    
    # 加载数据
    try:
        with open(pkl_path, 'rb') as f:
            dataset = pickle.load(f)
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return
    
    # 基本信息
    print(f"📊 基本信息:")
    print(f"   数据类型: {type(dataset)}")
    print(f"   样本数量: {len(dataset)}")
    print(f"   每个样本类型: {type(dataset[0]) if len(dataset) > 0 else 'N/A'}")
    print()
    
    if len(dataset) == 0:
        print("⚠️  数据集为空！")
        return
    
    # 检查第一个样本的键
    first_item = dataset[0]
    print(f"🔑 数据字段:")
    for key in first_item.keys():
        value = first_item[key]
        if isinstance(value, torch.Tensor):
            print(f"   - {key:25s}: Tensor {value.shape} ({value.dtype})")
        elif isinstance(value, list):
            print(f"   - {key:25s}: List (长度: {len(value)})")
        elif isinstance(value, (int, float)):
            print(f"   - {key:25s}: {type(value).__name__} = {value}")
        else:
            print(f"   - {key:25s}: {type(value).__name__}")
    print()
    
    # 统计信息
    print(f"📈 统计信息:")
    
    # 统计动作类型
    if 'motion_type' in first_item:
        motion_types = {}
        for item in dataset:
            mt = item.get('motion_type', 'Unknown')
            motion_types[mt] = motion_types.get(mt, 0) + 1
        print(f"   动作类型分布:")
        for mt, count in motion_types.items():
            print(f"      - {mt}: {count} 个样本")
    
    # 统计序列长度
    if 'coordinates' in first_item or 'original_seq_len' in first_item:
        seq_lens = []
        for item in dataset:
            if 'original_seq_len' in item:
                seq_lens.append(item['original_seq_len'])
            elif 'coordinates' in item:
                seq_lens.append(len(item['coordinates']))
        
        if seq_lens:
            print(f"   序列长度:")
            print(f"      - 最小: {min(seq_lens)} 帧")
            print(f"      - 最大: {max(seq_lens)} 帧")
            print(f"      - 平均: {sum(seq_lens)/len(seq_lens):.1f} 帧")
    
    # 统计标签数量
    if 'labels' in first_item:
        label_counts = [len(item['labels']) for item in dataset if 'labels' in item and item['labels']]
        if label_counts:
            print(f"   标签数量:")
            print(f"      - 最小: {min(label_counts)} 条")
            print(f"      - 最大: {max(label_counts)} 条")
            print(f"      - 平均: {sum(label_counts)/len(label_counts):.1f} 条")
    
    print()
    
    # 显示详细样本
    if show_details:
        print(f"{'='*80}")
        print(f"📋 样本详情 (前 {min(max_samples, len(dataset))} 个):")
        print(f"{'='*80}\n")
        
        for idx in range(min(max_samples, len(dataset))):
            item = dataset[idx]
            print(f"【样本 {idx + 1}】")
            print(f"  视频名称: {item.get('video_name', 'N/A')}")
            print(f"  动作类型: {item.get('motion_type', 'N/A')}")
            
            if 'coordinates' in item:
                coords = item['coordinates']
                print(f"  坐标数据: {coords.shape} (帧数×特征维度)")
                print(f"            min={coords.min():.3f}, max={coords.max():.3f}, mean={coords.mean():.3f}")
            
            if 'original_seq_len' in item:
                print(f"  原始长度: {item['original_seq_len']} 帧")
            
            if 'camera_view' in item:
                print(f"  摄像机视角: {item['camera_view']}")
            
            # 显示分段信息（如果有）
            segment_keys = ['aligned_start_frame', 'aligned_end_frame', 'aligned_seq_len',
                           'error_start_frame', 'error_end_frame', 'error_seq_len',
                           'gt_start_frame', 'gt_end_frame', 'gt_seq_len']
            segments = {k: item[k] for k in segment_keys if k in item}
            if segments:
                print(f"  分段信息:")
                for k, v in segments.items():
                    print(f"    - {k}: {v}")
            
            # 显示标签
            if 'labels' in item and item['labels']:
                print(f"  标签 ({len(item['labels'])} 条):")
                for i, label in enumerate(item['labels'][:2], 1):  # 只显示前2条
                    label_preview = label[:80] + '...' if len(label) > 80 else label
                    print(f"    [{i}] {label_preview}")
                if len(item['labels']) > 2:
                    print(f"    ... 还有 {len(item['labels']) - 2} 条标签")
            
            # 显示增强标签
            if 'augmented_labels' in item and item['augmented_labels']:
                print(f"  增强标签: {len(item['augmented_labels'])} 条")
            
            print()
    
    # 提供导出选项
    print(f"{'='*80}")
    print(f"💡 提示:")
    print(f"   - 使用 save_to_json=True 可以导出为 JSON 查看")
    print(f"   - 使用 max_samples 参数可以查看更多样本")
    print(f"{'='*80}\n")


def save_dataset_to_json(pkl_path, json_path):
    """
    将 PKL 数据集导出为 JSON（方便查看，但不包含 Tensor）
    """
    with open(pkl_path, 'rb') as f:
        dataset = pickle.load(f)
    
    # 转换为可序列化的格式
    json_data = []
    for item in dataset:
        json_item = {}
        for key, value in item.items():
            if isinstance(value, torch.Tensor):
                json_item[key] = {
                    "_type": "Tensor",
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "min": float(value.min()),
                    "max": float(value.max()),
                    "mean": float(value.mean())
                }
            elif isinstance(value, list):
                json_item[key] = value
            else:
                json_item[key] = value
        json_data.append(json_item)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 已导出到: {json_path}")


if __name__ == "__main__":
    # 查看 BX_test.pkl
    pkl_file = "dataset/BX_test.pkl"
    
    if not Path(pkl_file).exists():
        print(f"❌ 文件不存在: {pkl_file}")
        print(f"💡 请确保在项目根目录运行此脚本")
    else:
        # 查看数据集
        view_pkl_dataset(pkl_file, show_details=True, max_samples=3)
        
        # 可选：导出为 JSON
        # save_dataset_to_json(pkl_file, "dataset/BX_test_preview.json")
    
    print("\n" + "="*80)
    print("🔍 其他可查看的数据集:")
    print("   - dataset/BX_train.pkl")
    print("   - dataset/BX_standard.pkl")
    print("   - dataset/FS_test.pkl")
    print("   - dataset/FS_train.pkl")
    print("   - dataset/FS_standard.pkl")
    print("="*80)

