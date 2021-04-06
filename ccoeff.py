import cv2
import os
import numpy as np
from matplotlib import pyplot as plt


'''
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img_rgb = cv.imread('mario.png')
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
template = cv.imread('mario_coin.png',0)
w, h = template.shape[::-1]
res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
cv.imwrite('res.png',img_rgb)
'''


def argument(img):
    '''
    argument data to different direction
    1 image -> 6 images
    '''
    imgs = [img]
    for roate in [cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_90_CLOCKWISE,
                  cv2.ROTATE_180]:
        r_img = cv2.rotate(img, roate)
        imgs.append(r_img)
    imgs.append(img[:, ::-1, :])
    imgs.append(img[::-1, :, :])
    return imgs


def _template_matching_old(img, templates, return_ori=False):
    '''
    Drawbacks:
        1. Can only locate one ROI;
        2. Detect error by the similar between color of each pixel.
    Has change to `_template_matching()`
    '''
    H, W, C = img.shape
    h, w, c = templates[0].shape
    res = None
    for template in templates:
        if res is None:
            res = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
        else:
            # cv2.TM_SQDIFF_NORMED; TM_CCOEFF_NORMED
            tmp = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
            res = np.maximum(tmp, res)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    img_res = cv2.rectangle(np.zeros_like(img), top_left,
                            bottom_right, (1, 1, 1), -1)

    test_img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :]
    dis = [np.mean(test_img.astype(np.int32) - template.astype(np.int32), axis=2)
           for template in templates]

    if max([np.sum(d * d < 1) for d in dis]) < h * w * 0.7:
        img_res = np.zeros_like(img)
        # print("do not find")

    img_res = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)

    if return_ori:
        return img_res, res
    else:
        return img_res


def _template_matching(img, templates, threshold, return_ori=False):
    H, W, C = img.shape
    h, w, c = templates[0].shape
    res = None
    for template in templates:
        if res is None:
            res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        else:
            # cv2.TM_SQDIFF_NORMED; TM_CCOEFF_NORMED
            tmp = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED) # TM_CCORR_NORMED
            res = np.maximum(tmp, res)

    # Now we get `res`, the result (0-1 map) of matching
    # Detect the ROI by a fixed 0.8, the same as the paper

    loc = np.where(res >= threshold)
    img_res = np.zeros_like(img)
    for pt in zip(*loc[::-1]):
        img_res = cv2.rectangle(img_res, pt, (pt[0] + w, pt[1] + h), (1, 1, 1), -1)

    img_res = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
    if return_ori:
        return img_res, res
    else:
        return img_res


def template_matching(img, template_dir, threshold = 0.8, return_ori=False, vis=False, argumentation = True):
    if isinstance(img, str):
        img = cv2.imread(img)

#     img = img[:180, :, :]

    templates = []
    for t_file in os.listdir(template_dir):
        if t_file[-3:] != "png":
            continue
        template = cv2.imread(os.path.join(template_dir, t_file))
        if argumentation:
            templates.extend(argument(template))
        else:
            templates.extend([template])

    if return_ori:
        img_res, res = _template_matching(
            img, templates, threshold = threshold, return_ori=return_ori)
        if vis:
            img_res = cv2.cvtColor(img_res, cv2.COLOR_GRAY2BGR)
            added_image = cv2.addWeighted(img, 0.6, img_res*255, 0.4, 0)
            plt.subplot(121), plt.imshow(res, cmap='gray')
            plt.title('cv2.TM_CCORR_NORMED'), plt.xticks([]), plt.yticks([])

            plt.subplot(122), plt.imshow(added_image[:, :, ::-1])
            plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
            plt.show()
        return img_res, res
    else:
        img_res = _template_matching(img, templates)
        if vis:
            img_res = cv2.cvtColor(img_res, cv2.COLOR_GRAY2BGR)
            added_image = cv2.addWeighted(img, 0.6, img_res * 255, 0.4, 0)
            plt.imshow(added_image[:, :, ::-1])
            plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
            plt.show()
    return img_res


if __name__ == '__main__':
    template_dir = 'data/templates/monster-red/'
    img = 'data/source-train/00.png'
    img_res, _ = template_matching(img, template_dir,
                                vis=True, return_ori=True, argumentation=True)
    print(img_res.shape, np.max(img_res))
