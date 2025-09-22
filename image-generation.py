import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import re
from scipy.ndimage import gaussian_filter1d


def denoise_histogram(h, method="moving", window=150):
    if method == "moving":
        kernel = np.ones(window) / window
        return np.convolve(h, kernel, mode='same')
    elif method == "gaussian":
        return gaussian_filter1d(h, sigma=0.1)
    elif method == "median":
        from scipy.signal import medfilt
        return medfilt(h, kernel_size=5)
    else:
        return h  # 不去噪


# 参数设置
folder = "data/two50x50-1"
mn = 25000  # 你希望的动画帧数
group_size = 1  # 每 group_size 帧做一次平均，mn * group_size = 实际时间点数
c_dual = 1

# 解析图像尺寸
match = re.search(r"(\d+)x(\d+)", folder)
if match:
    row = int(match.group(1))
    col = int(match.group(2))
    print(f"✅ 检测到图像尺寸：{row}x{col}")
else:
    raise ValueError("❌ 无法从文件夹名中提取 row 和 col，请确保包含 'NxM' 格式。")

# 加载所有结果文件
all_files = sorted(glob.glob(f"{folder}/*.npy"))
all_files = [f for f in all_files if not f.endswith("backlight.npy")]
print(f"📂 共发现 {len(all_files)} 个文件")

# ✅ 加载背景光
backlight_path = f"{folder}/backlight.npy"
print(f"🌒 Loading backlight from: {backlight_path}")
backlight = np.load(backlight_path)

all_histograms = []
for f in all_files:
    data = np.load(f)
    if data.ndim == 1 and data.shape[0] == 4096:
        data = data[np.newaxis, :]  # 兼容 shape=(4096,) 的情况
    # ✅ 执行背景减除（并确保不为负）
    data = np.clip(data - backlight, 0, None)
    all_histograms.append(data)  # 每个 data 是 (N, 4096)

histogram_data = np.concatenate(all_histograms, axis=0)  # (总帧数, 4096)

# 去噪
print("🧹 开始对 histograms 去噪...")
histogram_data = np.array([denoise_histogram(h) for h in histogram_data])

total_pixels = row * col
n_frames = histogram_data.shape[0]
if n_frames % total_pixels != 0:
    raise ValueError("❌ 帧数不是像素个数的整数倍")
n_repeats = n_frames // total_pixels
print(f"🔁 每个像素共有 {n_repeats} 个 histogram")

# 初始化用于存储每个像素的时间序列向量的列表
pixel_vectors = []

# 计算总的时间点数：mn 是输出帧数，group_size 是每帧平均所用的时间点数量
total_timepoints = mn * group_size

# 遍历每个像素（假设图像有 total_pixels 个像素）
for i in range(total_pixels):
    trace = []  # 存储该像素的时间序列
    
    # 将该像素在每次重复实验中的数据按顺序拼接
    for j in range(n_repeats):
        # 对于每次重复，从 histogram_data 中获取该像素的时间序列
        # histogram_data 是 shape=(n_repeats * total_pixels, timepoints)
        idx = j * total_pixels + i  # 计算在 histogram_data 中的索引
        trace.append(histogram_data[idx])  # 添加该像素在这次重复实验中的数据

    # 拼接成一个完整的一维时间序列（shape = (n_repeats * timepoints,)）
    trace = np.concatenate(trace)

    # 确保时间序列长度足够，否则报错
    assert trace.shape[0] >= total_timepoints, f"数据不足：trace 长度 {trace.shape[0]} < {total_timepoints}"

    # 截取前 total_timepoints 长度（保证整除）
    trace = trace[:total_timepoints]

    # 分组平均：将时间序列 reshape 成 (mn, group_size) 形式，
    # 每 group_size 个数据作为一组，计算其均值，得到 mn 帧
    avg_vector = trace.reshape(mn, group_size).mean(axis=1)

    # 将该像素的 mn 帧平均值向量保存
    pixel_vectors.append(avg_vector)

# 将所有像素的平均向量堆叠成一个矩阵：shape = (total_pixels, mn)
# 每行是一个像素的 mn 帧向量，每列是一帧的所有像素值
value_matrix = np.array(pixel_vectors)

# 初始化累积图像，最终将所有帧累加成一个总图（用于可视化或分析）
total_image = np.zeros((row, col))

# 遍历每一帧
for frame_idx in range(mn):
    T = value_matrix[:, frame_idx]  # 获取该帧的所有像素值向量（长度 total_pixels）
    
    # 对该帧进行强度缩放，c_dual 是一个缩放因子（可以是 float 或向量）
    p_dual = c_dual * T

    # 将一维向量恢复为二维图像（row x col）
    image = p_dual.reshape(row, col)

    # 将该帧图像累加到最终的 total_image 上
    total_image += image


final_min = np.min(total_image)
final_max = np.max(total_image)

def generate_accumulated_image_no_decay(filename):
    total_image = np.zeros((row, col))
    clip_threshold = np.percentile(value_matrix, 100)  # 自动裁剪高亮点

    for frame_idx in range(mn):
        T = value_matrix[:, frame_idx]
        p_dual = c_dual * T
        image = p_dual.reshape(row, col)
        image = np.clip(image, 0, clip_threshold)
        total_image += image

    plt.figure()
    plt.imshow(total_image, cmap='gray')
    plt.axis('off')
    plt.title("Accumulated Image (No Decay, Clipped)")
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()


def generate_accumulated_image_main(filename):
    total_image = np.zeros((row, col))
    clip_threshold = np.percentile(value_matrix, 100)  # 可调整百分位裁剪

    p_main = np.ones((row * col,))  # shape = (total_pixels,)

    for frame_idx in range(mn):
        T = value_matrix[:, frame_idx]  # shape = (total_pixels,)
        
        # 计算加权内积：一个标量
        c_main = np.dot(p_main, T)  # shape = ()

        # 生成该帧的图像（仍然是 shape = (total_pixels,)）
        image = (c_main * p_main).reshape(row, col)  # 恢复成 (row, col)

        # 可选：裁剪图像防止高亮点影响
        image = np.clip(image, 0, clip_threshold)

        total_image += image  # 累积

    plt.figure()
    plt.imshow(total_image, cmap='gray')
    plt.axis('off')
    plt.title("Accumulated Image (Main)")
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()




# 运行
generate_accumulated_image_no_decay(f"{folder}_dual_image.png")
generate_accumulated_image_main(f"{folder}_main_image.png")
#create_frame_by_frame_animation_fixed_cmap(with_title=True, filename=f"{folder}_frame_by_frame_fixed.gif")
print("✅ 已生成图像！")
