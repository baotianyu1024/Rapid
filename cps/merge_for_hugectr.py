# 合并到一个数据集, 这个数据集可以直接给hugectr使用
import os
import struct
import argparse

# --- 1. 参数解析 ---

def parse_args():
    parser = argparse.ArgumentParser(description="Merge multiple device datasets into a unified HugeCTR binary file with interleaving.")

    # 路径参数
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='输入目录：存放 dataset_0.bin 到 dataset_N.bin 的地方')
    
    parser.add_argument('--output_file', type=str, required=True, 
                        help='输出文件路径：HugeCTR 训练实际读取的大文件')

    # 配置参数
    parser.add_argument('--num_devices', type=int, default=8, 
                        help='设备(GPU)数量 (默认: 8)')
    
    parser.add_argument('--per_gpu_batch_size', type=int, default=8192, 
                        help='单卡 Batch Size (默认: 8192)')
    
    parser.add_argument('--sample_size_bytes', type=int, default=912, 
                        help='单条样本的字节大小 (默认: 912, 即 4+13*4+214*4)')

    return parser.parse_args()

# --- 2. 合并脚本 ---

def merge_datasets_for_hugectr(args):
    input_dir = args.input_dir
    output_file = args.output_file
    num_devices = args.num_devices
    per_gpu_batch_size = args.per_gpu_batch_size
    sample_size_bytes = args.sample_size_bytes

    print(f"=== 开始合并数据集 ===")
    print(f"  输入目录: {input_dir}")
    print(f"  输出文件: {output_file}")
    print(f"  设备数量: {num_devices}")
    print(f"  单卡 Batch: {per_gpu_batch_size}")
    print(f"  样本大小: {sample_size_bytes} bytes")
    print(f"  策略: {num_devices} 卡交错合并 (Interleaved)")
    print("-" * 50)

    # 1. 打开所有源文件
    input_handles = []
    try:
        for i in range(num_devices):
            path = os.path.join(input_dir, f"dataset_{i}.bin")
            if not os.path.exists(path):
                raise FileNotFoundError(f"找不到输入文件: {path}")
            input_handles.append(open(path, "rb"))
        
        # 确保输出目录存在
        out_dir_path = os.path.dirname(output_file)
        if out_dir_path and not os.path.exists(out_dir_path):
            os.makedirs(out_dir_path)

        # 打开输出文件
        # 使用 Buffered I/O (默认为 -1) 提高写入性能
        with open(output_file, "wb") as f_out:
            
            batch_idx = 0
            # 计算每个设备每次读取的数据块大小 (Bytes per shard)
            bytes_per_shard = per_gpu_batch_size * sample_size_bytes
            
            while True:
                # 每一轮循环代表一个 Global Batch
                # 我们需要依次从 dataset_0 到 dataset_7 读取一个 shard
                
                shards = []
                all_eof = False
                any_eof = False
                
                for i in range(num_devices):
                    # 从对应设备的文件中读取一个 Batch 的数据
                    data = input_handles[i].read(bytes_per_shard)
                    
                    if len(data) == 0:
                        all_eof = True
                    elif len(data) < bytes_per_shard:
                        any_eof = True
                        print(f"[Warning] Device {i} 在 Batch {batch_idx} 数据不足 (只有 {len(data)} 字节).")
                    
                    shards.append(data)
                
                # 终止条件判断
                if all_eof:
                    print("\n所有文件读取完毕。")
                    break
                
                if any_eof:
                    print("\n检测到部分文件数据不足（对齐问题），停止合并以保证 Batch 完整性。")
                    break
                
                # 关键步骤：按顺序写入 (GPU 0 -> GPU 1 -> ... -> GPU 7)
                # 这样 HugeCTR Reader 计算 offset 时，GPU i 刚好读到 shards[i]
                for chunk in shards:
                    f_out.write(chunk)
                
                batch_idx += 1
                if batch_idx % 100 == 0:
                    print(f"已合并 {batch_idx} 个 Global Batches...", end='\r')

            total_samples = batch_idx * num_devices * per_gpu_batch_size
            print(f"\n合并完成! 总共生成 {batch_idx} 个 Global Batches.")
            print(f"总样本数: {total_samples}")
            
            if os.path.exists(output_file):
                print(f"输出文件大小: {os.path.getsize(output_file) / (1024**3):.2f} GB")

    except Exception as e:
        print(f"\n[Error] {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 关闭所有输入文件
        for f in input_handles:
            f.close()

# --- 3. 执行 ---

if __name__ == "__main__":
    args = parse_args()
    merge_datasets_for_hugectr(args)