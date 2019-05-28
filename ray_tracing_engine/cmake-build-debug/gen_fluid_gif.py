
# -*- coding: UTF-8 -*-

import imageio
import os


def create_gif(image_list, gif_name):

    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    # Save them as frames into a gif
    imageio.mimsave(gif_name, frames, 'GIF', duration=0.02)

    return


if __name__ == "__main__":
    image_list = []
    f_list = os.listdir(os.getcwd())
    # print f_list
    for i in f_list:
            # os.path.splitext():分离文件名与扩展名
        if os.path.splitext(i)[1] == '.png':
            image_list.append(i)

    image_list.sort()
    gif_name = 'created_gif.gif'
    create_gif(image_list, gif_name)
