# import os
# import re
# import numpy as np
# import matplotlib.pyplot as plt
# import networkx as nx
# from tqdm import tqdm
# import torch
# from torch_geometric.data import Data
#
# # ===========================
# # 路径配置
# # ===========================
# ROOT_DIR = r"E:\AL-LRR\data\ntu"
# SAVE_DIR = r"E:\AL-LRR\data\ntu_graph_pt"
# os.makedirs(SAVE_DIR, exist_ok=True)
#
# # ===========================
# # 定义骨架连接关系（NTU标准25关节）
# # ===========================
# NTU_SKELETON_EDGES = [
#     (1, 2), (2, 21), (21, 3), (3, 4), (21, 5), (5, 6),
#     (6, 7), (7, 8), (8, 22), (8, 23), (21, 9), (9, 10),
#     (10, 11), (11, 12), (12, 24), (12, 25), (1, 17),
#     (17, 18), (18, 19), (19, 20), (1, 13), (13, 14),
#     (14, 15), (15, 16)
# ]
#
# # ===========================
# # 解析 .skeleton 文件
# # ===========================
# def parse_skeleton_file(file_path):
#     """解析 NTU RGB+D .skeleton 文件，返回第1帧的25个关节坐标"""
#     with open(file_path, 'r') as f:
#         lines = f.readlines()
#
#     num_frames = int(lines[0].strip())
#     if num_frames == 0:
#         return None
#
#     idx = 1
#     # 只取第1帧
#     num_bodies = int(lines[idx].strip())
#     idx += 1
#     if num_bodies == 0:
#         return None
#
#     # 读取第一个人体
#     idx += 1  # 跳过 bodyID 行
#     num_joints = int(lines[idx].strip())
#     idx += 1
#     joints = []
#     for _ in range(num_joints):
#         joint_info = list(map(float, lines[idx].strip().split()))
#         joints.append(joint_info[:3])  # 取前三列 X,Y,Z
#         idx += 1
#
#     joints = np.array(joints[:25], dtype=np.float32)
#     return joints
#
# # ===========================
# # 绘制骨架图片
# # ===========================
# def render_skeleton_image(joints):
#     fig = plt.figure(figsize=(3, 3))
#     xs, ys = joints[:, 0], joints[:, 1]
#     for i, j in NTU_SKELETON_EDGES:
#         if i <= len(joints) and j <= len(joints):
#             plt.plot([xs[i - 1], xs[j - 1]], [ys[i - 1], ys[j - 1]], 'b-', linewidth=2)
#     plt.scatter(xs, ys, c='r', s=10)
#     plt.axis('off')
#     plt.gca().invert_yaxis()
#     plt.tight_layout(pad=0)
#     fig.canvas.draw()
#     img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#     img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#     plt.close(fig)
#     return img
#
# # ===========================
# # 主函数
# # ===========================
# def process_skeleton_files(root_dir, save_dir):
#     files = [f for f in os.listdir(root_dir) if f.endswith(".skeleton")]
#     print(f"📁 共发现 {len(files)} 个 skeleton 文件")
#
#     # 仅保留 A008 和 A009
#     selected_files = [f for f in files if "A008" in f or "A009" in f]
#     print(f"✅ 选中 {len(selected_files)} 个 A008/A009 文件")
#
#     vis_count = 0  # 用于显示前两张图片
#
#     for file_name in tqdm(selected_files, desc="⏳ 处理文件"):
#         file_path = os.path.join(root_dir, file_name)
#
#         joints = parse_skeleton_file(file_path)
#         if joints is None or joints.shape[0] < 25:
#             print(f"⚠️ 跳过无效文件: {file_name}")
#             continue
#
#         # 构建骨架图
#         G = nx.Graph()
#         for i in range(len(joints)):
#             G.add_node(i, x=joints[i, 0], y=joints[i, 1], z=joints[i, 2])
#         G.add_edges_from(NTU_SKELETON_EDGES)
#         adj = nx.to_numpy_array(G)
#         features = joints
#
#         # 转成 PyG 格式
#         edge_index = np.array(np.nonzero(adj))
#         edge_index = torch.tensor(edge_index, dtype=torch.long)
#         x = torch.tensor(features, dtype=torch.float)
#
#         # 动作标签
#         y = torch.tensor(0 if "A008" in file_name else 1, dtype=torch.long)
#
#         # 渲染骨架图片
#         try:
#             img_array = render_skeleton_image(joints)
#         except Exception as e:
#             print(f"⚠️ 渲染失败: {file_name} - {e}")
#             continue
#
#         # ✅ 可视化前两张渲染图
#         if vis_count < 2:
#             plt.imshow(img_array)
#             plt.title(f"{file_name} | Label={int(y)}")
#             plt.axis('off')
#             plt.show()
#             vis_count += 1
#
#         # 构建图数据对象
#         data = Data(x=x, edge_index=edge_index, y=y)
#         data.image = torch.tensor(img_array)
#
#         # 保存为 .pt
#         save_path = os.path.join(save_dir, file_name.replace(".skeleton", ".pt"))
#         torch.save(data, save_path)
#
#     print(f"🎉 所有数据已保存至: {save_dir}")
#
#
# # ===========================
# # 执行主程序
# # ===========================
# if __name__ == "__main__":
#     process_skeleton_files(ROOT_DIR, SAVE_DIR)

#
# import torch
#
# # 文件路径
# file_path = r"E:\AL-LRR\data\ntu_graph_pt\S001C001P001R001A008.pt"
#
# # 读取
# data = torch.load(file_path, map_location="cpu")
#
# print("数据类型:", type(data))
#
# # 如果是字典
# if isinstance(data, dict):
#     print("\n==== 字典内容 ====")
#     for key in data:
#         value = data[key]
#         print(f"\nKey: {key}")
#         print("  类型:", type(value))
#
#         if isinstance(value, torch.Tensor):
#             print("  shape:", value.shape)
#             print("  dtype:", value.dtype)
#         else:
#             print("  内容示例:", str(value)[:200])
#
# # 如果是张量
# elif isinstance(data, torch.Tensor):
#     print("\n==== Tensor 信息 ====")
#     print("shape:", data.shape)
#     print("dtype:", data.dtype)
#     print("前5个元素:\n", data.flatten()[:5])
#
# # 如果是列表
# elif isinstance(data, list):
#     print("\n==== List 信息 ====")
#     print("长度:", len(data))
#     print("前1个元素类型:", type(data[0]))
#
# else:
#     print("\n未知结构，内容预览:")
#     print(str(data)[:500])
#
#
# #打印第一个文件内容
# import torch
# print(torch.load("E:\\AL-LRR\\data\\ntu_graph_pt\\S001C001P001R001A008.pt"))
