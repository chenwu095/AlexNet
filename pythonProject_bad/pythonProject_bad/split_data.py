import os
from shutil import copy
import random


def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)


file_path = '/Users/magnet/PycharmProjects/pythonProject/dataset'
flower_class = [cla for cla in os.listdir(file_path)]

mkfile('data/train')
for cla in flower_class:
    mkfile('data/train/' + cla)

mkfile('data/val')
for cla in flower_class:
    mkfile('data/val/' + cla)

split_rate = 0.2

for cla in flower_class:
    cla_path = file_path + '/' + cla + '/'  
    images = os.listdir(cla_path)  
    num = len(images)
    eval_index = random.sample(images, k=int(num * split_rate))  
    for index, image in enumerate(images):
        if image in eval_index:
            image_path = cla_path + image
            new_path = 'data/val/' + cla
            copy(image_path, new_path) 

        else:
            image_path = cla_path + image
            new_path = 'data/train/' + cla
            copy(image_path, new_path)
        print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")  # processing bar
    print()

print("processing done!")
