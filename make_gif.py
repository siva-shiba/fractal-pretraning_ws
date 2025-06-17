from fractal_learning.fractals import ifs
import numpy as np
import imageio
from tqdm import tqdm

# パラメータ
image_size = 500
frame_count = 300
points_per_frame = 300_000
output_path = "fractal_morph_multi.gif"

def interpolate_system(system_a, system_b, t):
    system_a = np.array(system_a)
    system_b = np.array(system_b)
    return (1 - t) * system_a + t * system_b

# start, mid1, mid2, mid3, end
systems = [ifs.sample_system((8, 9)) for _ in range(10)]


segments = len(systems) - 1
frames_per_segment = frame_count // segments
frames = []

for seg in range(segments):
    sys_a = systems[seg]
    sys_b = systems[seg + 1]
    
    for i in tqdm(range(frames_per_segment), desc=f"Segment {seg+1}/{segments}", position=0, leave=True):
        t = i / frames_per_segment
        interp_system = np.array(interpolate_system(sys_a, sys_b, t))

        points = ifs.iterate(interp_system, points_per_frame)
        gray_image = ifs.render(points, s=image_size, binary=False, patch=False)

        # カラー化 & 変換
        color_image = ifs.colorize(gray_image)
        rgb_image = color_image[..., ::-1]  # BGR→RGB
        rgb_image_uint8 = rgb_image.clip(0, 255).astype(np.uint8)

        frames.append(rgb_image_uint8)


# GIF保存
imageio.mimsave(output_path, frames, duration=0.15, loop=0)
print(f"GIF saved to: {output_path}")
