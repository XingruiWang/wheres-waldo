import cv2
import os
import numpy as np
from matplotlib import pyplot as plt


def argument(img):
    imgs = []
    for roate in [cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_90_CLOCKWISE,
                  cv2.ROTATE_180]:
        r_img = cv2.rotate(img, roate)
        imgs.append(r_img)
    imgs.append(img[:, ::-1, :])
    imgs.append(img[::-1, :, :])
    return imgs


def _template_matching(img, templates, return_ori=False):
    H, W, C = img.shape
    h, w, c = templates[0].shape
    res = None
    for template in templates:
        if res is None:
            res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        else:
            tmp = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED) # cv2.TM_SQDIFF_NORMED; TM_CCOEFF_NORMED
            res = np.maximum(tmp, res)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    img_res = cv2.rectangle(np.zeros_like(img), top_left,
                            bottom_right, (1, 1, 1), -1)

    test_img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :]


    dis = [np.mean(test_img.astype(np.int32) - template.astype(np.int32), axis=2) for template in templates]

    if max([np.sum(d * d < 1) for d in dis]) < h * w * 0.7:
        img_res = np.zeros_like(img)
        # print("do not find")

    img_res = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)

    if return_ori:
        return img_res, res
    else:
        return img_res


def template_matching(img, template_dir, return_ori=False, vis=False):
    if isinstance(img, str):
        img = cv2.imread(img)

    img = img[:180, :, :]

    templates = []
    for t_file in os.listdir(template_dir):
        template = cv2.imread(os.path.join(template_dir, t_file))
        templates.extend(argument(template))
    if return_ori:
        img_res, res = _template_matching(
            img, templates, return_ori=return_ori)
        if vis:
            img_res = cv2.cvtColor(img_res, cv2.COLOR_GRAY2BGR)
            added_image = cv2.addWeighted(img, 0.6, img_res*255, 0.4, 0)
            plt.subplot(121), plt.imshow(res, cmap='gray')
            plt.title('cv2.TM_CCOEFF'), plt.xticks([]), plt.yticks([])

            plt.subplot(122), plt.imshow(added_image[:, :, ::-1])
            plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
            plt.show()
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
    template_dir = 'data/templates/pacman/'
    img = 'data/source/02.png'
    img_res = template_matching(img, template_dir,
                                vis=False, return_ori=False)
    print(img_res.shape, np.max(img_res))
