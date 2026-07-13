import argparse
import os
import sys
import json

# 定义五个模型的基础名称
MODELS = ["MeshEncoder", "G2MAggregate", "M2MProcessor", "M2GInteraction", "ForceField"]
JSON_NAME = "GraphCastInfer.json"

def parse_args():
    parser = argparse.ArgumentParser(description="TensorRT Python 批量编译、显存统计并更新 JSON 脚本")
    
    # 互斥组：精度参数只能选择一种
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--fp16', action='store_true', help='启用 FP16 模式')
    group.add_argument('--bf16', action='store_true', help='启用 BF16 模式')
    group.add_argument('--noTF32', action='store_true', help='禁用 TF32 模式（默认开启 TF32）')
    
    # ONNX 文件和 JSON 所在的路径参数
    parser.add_argument('--onnx-dir', type=str, default=".", help='ONNX 模型与 JSON 文件所在的目录路径 (默认: 当前目录)')
    
    # 对应 trtexec 的 --maxAuxStreams 
    parser.add_argument('--maxAuxStreams', type=int, default=0, help='最大辅助流数量 (默认: 0)')
    
    return parser.parse_args()

def build_engines(args):
    """使用 TensorRT Python API 批量编译模型"""
    print("=" * 60)
    print("开始使用 TensorRT Python API 编译引擎...")
    print(f"工作目录: {os.path.abspath(args.onnx_dir)}")
    print("=" * 60)

    try:
        import tensorrt as trt
        trt.init_libnvinfer_plugins(None, "")
    except ImportError:
        print("[错误] 未检测到 Python tensorrt 库，请先安装: pip install tensorrt", file=sys.stderr)
        return False, {}

    logger = trt.Logger(trt.Logger.INFO)
    vram_report = {}

    for model in MODELS:
        onnx_file = os.path.join(args.onnx_dir, f"{model}.onnx")
        engine_file = os.path.join(args.onnx_dir, f"{model}.engine")
        
        if not os.path.exists(onnx_file):
            print(f"\n[错误] 找不到 ONNX 文件: {onnx_file}，跳过后续编译。", file=sys.stderr)
            return False, {}

        print(f"\n正在编译: {onnx_file} -> {engine_file}")

        # 1. 创建 Builder, Network 和 Parser
        builder = trt.Builder(logger)
        explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(explicit_batch)
        parser = trt.OnnxParser(network, logger)

        # 2. 解析 ONNX 模型
        with open(onnx_file, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                print(f"[错误] 解析 ONNX 失败: {onnx_file}")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return False, {}

        # 3. 配置编译参数 (Config)
        config = builder.create_builder_config()
        
        if hasattr(config, "max_aux_streams"):
            config.max_aux_streams = args.maxAuxStreams

        # 配置精度控制（ForceField 模型不配置精度）
        if model != 'ForceField':
            if args.noTF32:
                config.clear_flag(trt.BuilderFlag.TF32)
            
            if args.fp16:
                if builder.platform_has_fast_fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
                else:
                    print("[警告] 当前硬件平台不支持 FP16 加速。")
            elif args.bf16:
                # 检查当前 TRT 版本是否支持 BF16 枚举
                if hasattr(trt.BuilderFlag, "BF16"):
                    config.set_flag(trt.BuilderFlag.BF16)
                else:
                    print("[警告] 当前 TensorRT 版本太低，其 BuilderFlag 不支持 BF16 加速。")
                    
        # 4. 构建并序列化 Engine
        print("正在构建和优化 CUDA Engine (这可能需要几分钟)...")
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            print(f"[错误] 编译模型 {model} 失败。")
            return False, {}

        # 5. 保存序列化后的文件到硬盘
        with open(engine_file, 'wb') as f:
            f.write(plan)
        print(f"成功保存 Engine 至: {engine_file}")

        # 6. 反序列化获取运行时显存大小 (Bytes)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(plan)
        if engine:
            # 存入字典，Key 使用文件名如 "MeshEncoder.engine"，方便后续与 JSON 匹配
            vram_report[f"{model}.engine"] = engine.device_memory_size

    return True, vram_report

def update_json_file(onnx_dir, vram_report):
    """读取、更新并写回 GraphCastInfer.json"""
    json_path = os.path.join(onnx_dir, JSON_NAME)
    
    if not os.path.exists(json_path):
        print(f"\n[警告] 未在目录中找到 {JSON_NAME}，跳过 JSON 更新。")
        return

    print(f"\n正在更新配置文件: {json_path} ...")
    
    try:
        # 1. 读取原 JSON 内容
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 2. 遍历引擎列表并匹配更新
        updated_count = 0
        if "engines" in data and isinstance(data["engines"], list):
            for engine_info in data["engines"]:
                engine_file = engine_info.get("file")
                if engine_file in vram_report:
                    # 更新为最新的字节数
                    old_len = engine_info.get("contextlen", "None")
                    new_len = vram_report[engine_file]
                    engine_info["contextlen"] = new_len
                    print(f" - 已将 {engine_file} 的 contextlen 从 {old_len} 更新为 {new_len}")
                    updated_count += 1

        # 3. 写回文件，保持缩进（indent=2 可根据习惯改为 4）
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        print(f"JSON 成功更新完毕！共修改了 {updated_count} 处引擎配置。")

    except Exception as e:
        print(f"[错误] 更新 JSON 过程中发生异常: {e}", file=sys.stderr)

def print_vram_table(vram_report):
    """打印最终的显存统计表格"""
    print("\n" + "=" * 60)
    print(" 最终的 GPU 显存占用统计 (VRAM Usage) ")
    print("=" * 60)
    print(f"{'Engine 文件名':<25} | {'运行时显存占用 (Device Memory)':<30}")
    print("-" * 60)
    
    for filename, mem_bytes in vram_report.items():
        mem_gb = mem_bytes / (1024 ** 3)
        print(f"{filename:<25} | {mem_bytes:<10} bytes ({mem_gb:.4f} GB)")
        
    print("=" * 60)

if __name__ == "__main__":
    arguments = parse_args()
    
    # 1. 一键运行编译并直接获取显存结果
    success, report = build_engines(arguments)
    
    if success:
        # 2. 打印显存表格
        print_vram_table(report)
        # 3. 自动更新同目录下的 JSON 配置文件
        update_json_file(arguments.onnx_dir, report)
    else:
        print("\n[错误] 由于编译中途出错，停止运行，JSON 未做任何修改。", file=sys.stderr)
        sys.exit(1)
