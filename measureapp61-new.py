import cv2
import numpy as np
import time
from tqdm import tqdm
from module import PyPicoharp
import os

# --- 设置显示器分辨率（副屏） ---
screen_width = 1280
screen_height = 800

rows, cols = 50,50

# --- 计算格子的最大尺寸，使每个格子是正方形 ---
max_cell_width = screen_width // cols
max_cell_height = screen_height // rows
cell_size = min(max_cell_width, max_cell_height)

grid_width = cols * cell_size
grid_height = rows * cell_size

x_offset = (screen_width - grid_width) // 2
y_offset = (screen_height - grid_height) // 2

def create_output_folder(path):
    index = 2
    folder_name = path
    created = False
    while not created:
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)
            created = True
            return folder_name
        folder_name = path + str(index)
        index += 1

def save_results(path_output_folder, index, results):
    file_path = os.path.join(path_output_folder, '%04d.npy' % (index))
    np.save(file_path, results)
    return (index + 1), [], 0

def main(path_output='./data/results'):
    path_output_folder = create_output_folder(path_output)

    print("PHLib version " + PyPicoharp.lib_version())
    dev_id = PyPicoharp.get_device_id()
    device = PyPicoharp.picoharp()
    device.device_id = dev_id

    device.mode = 0  # histogramming mode
    device.binning = 0
    device.offset = 0
    device.acquisition_time_ms = 20000
    device.sync_divider = 1  # 1 for none

    device.CFD_zero_cross0 = 10
    device.CFD_level0_mV = 50
    device.CFD_zero_cross1 = 10
    device.CFD_level1_mV = 50

    if not device.open():
        print(device.get_error_message())
        return

    cv2.namedWindow("Grid", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Grid", 1920, 0)  # 根据实际副屏位置调整
    cv2.setWindowProperty("Grid", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("Preparing for backlight measurement...")
    print("Showing all black screen and measuring background light...\a")

    # === 显示全黑并测量背景光 ===
    black_img = np.zeros((screen_height, screen_width), dtype=np.uint8)
    cv2.imshow("Grid", black_img)
    cv2.waitKey(10)

    if not device.start_measurement():
        print("❌ Failed to start background measurement:", device.get_error_message())
        return

    backlight_hist = np.asarray(device.get_histogram())
    np.save(os.path.join(path_output_folder, "backlight.npy"), backlight_hist)
    print("✅ Background measurement saved to backlight.npy")

    time.sleep(1)  # 暂停一秒

    # === 正式开始测量 ===
    print("Measuring starting in 1 second. Check rows and cols!! Check laser power!")
    print('\a')
    time.sleep(1)

    histogram_max = 1500
    histogram_index = 0
    save_index = 0
    results = []

    total_frames = rows * cols
    frame = 0

    pbar = tqdm(range(total_frames))
    for _ in pbar:

        # 创建黑色图像
        img = np.zeros((screen_height, screen_width), dtype=np.uint8)

        # 计算当前格子位置并点亮
        r = frame // cols
        c = frame % cols
        y1, y2 = r * cell_size + y_offset, (r + 1) * cell_size + y_offset
        x1, x2 = c * cell_size + x_offset, (c + 1) * cell_size + x_offset
        img[y1:y2, x1:x2] = 255
        
        # 显示当前帧的网格
        cv2.imshow("Grid", img)
        key = cv2.waitKey(10)
        if key == 27:  # ESC退出
            break

        # 进行PicoHarp测量
        if not device.start_measurement():
            print(device.get_error_message())
            break

        if device.is_overflow:
            print("Overflow detected!")

        hist = np.asarray(device.get_histogram())

        pbar.set_description(f"Max hist: {np.max(hist)}")

        histogram_index += 1
        results.append(hist)
        

        if histogram_index == histogram_max:
            save_index, results, histogram_index = save_results(path_output_folder, save_index, results)

        frame = (frame + 1) % total_frames

    # 保存剩余数据
    if histogram_index > 0:
        save_index, results, histogram_index = save_results(path_output_folder, save_index, results)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
