import random
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.affinity import rotate, translate


def get_patches(valid_position, patch_size):
    patch_width, patch_height = patch_size
    patch_bboxes = []
    patch_polygons = []

    for cx, cy in valid_position:
        angle = random.uniform(0, 2 * np.pi)
        patch_obb = create_obb(cx, cy, patch_width, patch_height, angle)
        patch_polygons.append(patch_obb)
        patch_bboxes.append([cx, cy, patch_width, patch_height, angle])

    # Convert patch_bboxes to tensor
    patch_bboxes_tensor = torch.tensor(patch_bboxes, dtype=torch.float32)
    return patch_bboxes_tensor, patch_polygons


def main():
    # 示例用法
    img_size = (500, 500)
    obbs = [
        create_obb(200, 200, 50, 100, np.pi/4),
        create_obb(300, 300, 80, 40, -np.pi/6),
        create_obb(400, 400, 60, 120, np.pi/3)
    ]
    patch_size = (30, 30)
    number = 5

    positions = find_patch_positions(img_size, obbs, patch_size, number)
    print("Valid positions for patches:", positions)

def show_polygon_obbs(obbs,fake_obbs,i):
    def plot_polygon(polygon, ax, color='blue'):
        x, y = polygon.exterior.xy
        ax.plot(x, y, color=color)
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for obb in obbs:
        plot_polygon(obb, ax, color='green')
    for fake_obb in fake_obbs:
        plot_polygon(fake_obb, ax, color='red')

    ax.set_xlim(0, 2720)
    ax.set_ylim(0, 1536)
    ax.invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(f'test{i}_obb.png')


if __name__ == "__main__":
    import torch
    bboxes_tensor = torch.tensor([  [ 2.0820e+02,  1.3465e+03,  2.8000e+02,  1.3918e+02, -1.4898e+00],
                                    [ 3.7188e+02,  1.2906e+03,  2.4431e+02,  1.2753e+02, -1.5534e+00],
                                    [ 6.2546e+02,  1.2977e+03,  2.3699e+02,  1.0640e+02,  5.3840e-02],
                                    [ 6.5167e+02,  1.1326e+03,  2.2342e+02,  1.0790e+02,  6.9949e-02],
                                    [ 5.9429e+02,  9.9733e+02,  2.1760e+02,  1.0912e+02,  3.2668e-02],
                                    [ 5.5675e+02,  8.3300e+02,  3.7438e+02,  1.4908e+02, -7.2118e-02],
                                    [ 7.0503e+02,  6.3467e+02,  2.6920e+02,  1.2045e+02, -1.5788e-02],
                                    [ 1.6295e+03,  1.5499e+02,  2.7707e+02,  1.3049e+02, -3.6057e-01],
                                    [ 1.6390e+03,  3.2632e+02,  2.7418e+02,  1.1693e+02, -3.3324e-01],
                                    [ 1.6107e+03,  5.1708e+02,  2.6603e+02,  1.2636e+02, -3.4877e-01],
                                    [ 1.6079e+03,  6.7008e+02,  2.4163e+02,  1.0263e+02, -3.6084e-01],
                                    [ 1.5633e+03,  8.8400e+02,  2.7567e+02,  1.2813e+02, -3.1362e-01],
                                    [ 1.5789e+03,  1.1836e+03,  2.6645e+02,  1.2326e+02,  1.0638e-02],
                                    [ 1.5930e+03,  1.4903e+03,  2.8418e+02,  9.0764e+01, -3.9979e-02]
                                ])
    obbs = create_obb_tensor(bboxes_tensor)
    img_size = (2720,1536)
    patch_size = (128, 128)
    number = 2


    for i in range(10):
        positions = find_patch_positions(img_size, obbs, patch_size, number)
        patch_bboxes_tensor, patch_polygons = get_patches(positions, patch_size)
        show_polygon_obbs(obbs,patch_polygons,i)