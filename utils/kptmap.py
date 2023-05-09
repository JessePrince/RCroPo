import math
import numpy as np

def _add_gaussian(keypoint_map, x, y, stride, sigma):
    n_sigma = 4
    tl = [int(x - n_sigma * sigma), int(y - n_sigma * sigma)]
    tl[0] = max(tl[0], 0)
    tl[1] = max(tl[1], 0)

    br = [int(x + n_sigma * sigma), int(y + n_sigma * sigma)]
    map_h, map_w = keypoint_map.shape
    br[0] = min(br[0], map_w * stride)
    br[1] = min(br[1], map_h * stride)

    shift = stride / 2 - 0.5
    for map_y in range(tl[1] // stride, br[1] // stride):
        for map_x in range(tl[0] // stride, br[0] // stride):
            d2 = (map_x * stride + shift - x) * (map_x * stride + shift - x) + \
                (map_y * stride + shift - y) * (map_y * stride + shift - y)
            exponent = d2 / 2 / sigma / sigma
            if exponent > 4.6052:  # threshold, ln(100), ~0.01
                continue
            keypoint_map[map_y, map_x] += math.exp(-exponent)
            if keypoint_map[map_y, map_x] > 1:
                keypoint_map[map_y, map_x] = 1
                    
def _generate_keypoint_maps(image_w, image_h, label, stride, sigma):
    n_keypoints = 18
    n_rows = image_h
    n_cols = image_w
    keypoint_maps = np.zeros(shape=(n_keypoints + 1,
                                    n_rows // stride, n_cols // stride), dtype=np.float32)  # +1 for bg

    '''print('xxxx',label)'''
    for keypoint_idx in range(n_keypoints):
        keypoint = label[keypoint_idx]
        if keypoint[2] <= 1:
            _add_gaussian(keypoint_maps[keypoint_idx], keypoint[0], keypoint[1], stride, sigma)

    keypoint_maps[-1] = 1 - keypoint_maps.max(axis=0)
    return keypoint_maps

