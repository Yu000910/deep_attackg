import numpy as np
import pandas as pd
import os

# 你的数据集路径
FILE_PATH = "./D_BEDR.npz"

def inspect_npz(path):
    print(f">>> 📂 Loading Dataset from: {path}...")
    
    if not os.path.exists(path):
        print(f"❌ Error: File not found at {path}")
        return

    try:
        # allow_pickle=True 是为了加载非数值型数据（如文本字符串）
        data = np.load(path, allow_pickle=True)
        keys = list(data.keys())
        print(f"✅ Loaded successfully! Keys found: {keys}")
        
        # 1. 基础结构分析
        text_key = None
        label_key = None
        
        for key in keys:
            arr = data[key]
            print(f"\n--- [Key: '{key}'] ---")
            print(f"   Shape: {arr.shape}")
            print(f"   Dtype: {arr.dtype}")
            
            # 尝试打印前一个非空样本，看看长什么样
            if len(arr) > 0:
                sample = arr[0]
                # 如果是字节串，尝试解码
                if isinstance(sample, bytes):
                    try:
                        print(f"   Sample[0] (decoded): {sample.decode('utf-8')[:100]}...")
                    except:
                        print(f"   Sample[0]: {sample}")
                else:
                    print(f"   Sample[0]: {str(sample)[:100]}...")

            # 简单的启发式规则来猜测哪个是文本，哪个是标签
            # 通常文本是 Object/String 类型，标签可能是 Int 或 Object
            if arr.ndim == 1 and (np.issubdtype(arr.dtype, np.object_) or np.issubdtype(arr.dtype, np.str_)):
                # 如果内容看起来像句子（长度>20），大概率是文本
                if len(str(arr[0])) > 20:
                    text_key = key
                else:
                    # 短字符串可能是标签（如 'T1059'）
                    if not label_key: label_key = key
            elif np.issubdtype(arr.dtype, np.integer):
                label_key = key

        # 2. 标签分布深度分析
        # 如果我们猜到了标签列（或者用户手动指定，比如 keys 里有 'y' 或 'labels'）
        # 常见的名字: 'y', 'labels', 'label', 'Y', 'target'
        potential_label_keys = [k for k in keys if k.lower() in ['y', 'label', 'labels', 'target', 'targets']]
        if potential_label_keys:
            label_key = potential_label_keys[0]
        
        if label_key:
            print(f"\n" + "="*40)
            print(f"📊 Label Distribution Analysis (Target Key: '{label_key}')")
            print("="*40)
            
            labels = data[label_key]
            
            # 如果是 One-Hot (二维数组)，转成 Index
            if labels.ndim > 1 and labels.shape[1] > 1:
                print("   Note: Detected One-Hot encoding. Converting to indices...")
                labels = np.argmax(labels, axis=1)
            
            # 统计
            # 将 numpy array 转为 pandas Series 方便统计
            s = pd.Series(labels)
            counts = s.value_counts()
            
            print(f"   Total Samples: {len(s)}")
            print(f"   Unique Classes: {len(counts)}")
            print("-" * 40)
            print(f"   📈 Most Frequent (Top 5):")
            print(counts.head(5).to_string())
            print("-" * 40)
            print(f"   📉 Least Frequent (Bottom 5):")
            print(counts.tail(5).to_string())
            print("-" * 40)
            
            # 统计不平衡度
            max_c = counts.max()
            min_c = counts.min()
            mean_c = counts.mean()
            median_c = counts.median()
            
            print(f"   Max samples per class: {max_c}")
            print(f"   Min samples per class: {min_c}")
            print(f"   Mean samples: {mean_c:.2f}")
            print(f"   Median samples: {median_c:.2f}")
            print(f"   Imbalance Ratio (Max/Min): {max_c/min_c:.2f}x")
            
            if max_c / min_c > 10:
                print("\n⚠️ WARNING: Severe Class Imbalance Detected!")
                print("   Suggestion: Use Weighted Loss or Oversampling during training.")
        else:
            print("\n⚠️ Could not automatically identify the Label key. Please check the 'Keys' output above.")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    inspect_npz(FILE_PATH)