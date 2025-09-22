import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
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
folder = "data/bunnyfinal50x50"
mn = 8000  # 你希望的动画帧数
group_size = 1  # 每 group_size 帧做一次平均，mn * group_size = 实际时间点数
c_dual = 1.4

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
    if data.ndim == 1 and data.shape[0] == 25000:
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

# 构建每个像素的时间序列，拼接后做分组平均
pixel_vectors = []
total_timepoints = mn * group_size
for i in range(total_pixels):
    trace = []
    for j in range(n_repeats):
        idx = j * total_pixels + i
        trace.append(histogram_data[idx])
    trace = np.concatenate(trace)
    assert trace.shape[0] >= total_timepoints, f"数据不足：trace 长度 {trace.shape[0]} < {total_timepoints}"
    trace = trace[:total_timepoints]
    # 分组平均，得到 mn 帧
    avg_vector = trace.reshape(mn, group_size).mean(axis=1)
    pixel_vectors.append(avg_vector)

value_matrix = np.array(pixel_vectors)  # shape = (row*col, mn)

# 计算最终累积图的范围
total_image = np.zeros((row, col))
for frame_idx in range(mn):
    T = value_matrix[:, frame_idx]
    p_dual = c_dual * T
    image = p_dual.reshape(row, col)
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


def create_frame_by_frame_animation_fixed_cmap(with_title, filename):
    # 添加整体统计信息
    print("🧪 Debug: value_matrix 全局统计信息")
    print(f" - Mean: {np.mean(value_matrix):.5f}")
    print(f" - Std : {np.std(value_matrix):.5f}")
    print(f" - Min : {np.min(value_matrix):.5f}")
    print(f" - Max : {np.max(value_matrix):.5f}")

    # 使用分位数设置色阶，更稳健
    vmin = np.percentile(value_matrix, 0)
    vmax = np.percentile(value_matrix, 100)
    print(f"🎨 使用 vmin/vmax = ({vmin:.5f}, {vmax:.5f})")

    threshold = 0.4  # 设定“太黑”的阈值

    # 预筛选有效帧索引
    valid_frames = []
    for idx in range(mn):
        T = value_matrix[:, idx]
        p_dual = c_dual * T
        image = p_dual.reshape(row, col)
        # 过滤噪声
        image[image < threshold] = 0
        if image.max() > threshold:
            valid_frames.append(idx)
    print(f"筛选后有效帧数: {len(valid_frames)} / {mn}")

    fig, ax = plt.subplots()
    img = ax.imshow(np.zeros((row, col)), cmap='gray', vmin=vmin, vmax=vmax)
    if not with_title:
        ax.axis('off')

    def update(frame_idx):
        real_idx = valid_frames[frame_idx]
        T = value_matrix[:, real_idx]
        p_dual = c_dual * T
        image = p_dual.reshape(row, col)

        # 🌈 Apply gamma correction
        gamma = 1  # 可以调试这个值
        image_norm = np.clip((image - vmin) / (vmax - vmin), 0, 1)
        image_gamma = image_norm ** gamma
        image = image_gamma * (vmax - vmin) + vmin

        '''
        # 对数变换调整
        image_shifted = image - vmin
        image_shifted[image_shifted < 1e-6] = 1e-6  # 避免 log(0)
        image_log = np.log(image_shifted)
        image_norm = (image_log - image_log.min()) / (image_log.max() - image_log.min())
        image = image_norm * (vmax - vmin) + vmin
        '''

        # Debug: 打印部分帧的图像统计值（每隔50帧打印一次）
        if real_idx % 50 == 0 or real_idx == 0 or real_idx == mn - 1:
            print(f"  🔄 Frame {real_idx}:")
            print(f"     - image.min(): {image.min():.5f}")
            print(f"     - image.max(): {image.max():.5f}")
            print(f"     - image.mean(): {image.mean():.5f}")

        img.set_data(image.copy())

        if with_title:
            ax.set_title(f"Frame {real_idx}", fontsize=10)

        return [img]

    ani = animation.FuncAnimation(fig, update, frames=len(valid_frames))
    #ani.save(filename, writer='pillow')
    ani.save(filename, writer=FFMpegWriter(fps=40))
    plt.close(fig)



# 运行
#generate_accumulated_image_no_decay(f"{folder}_dual_image.png")
create_frame_by_frame_animation_fixed_cmap(with_title=True, filename=f"{folder}_frame_by_frame_fixed.mp4")
print("✅ 已生成动画！")
