from os import listdir
import random
import scipy.ndimage as ndimage
from scipy.misc import imresize
import numpy as np

def read_legend(legend_name):
    with open(legend_name) as file:
        filename_list = [x.strip() for x in file.readlines()]
    return filename_list

def get_info(dataset_path):
    info = {}
    problem_sets = read_legend('{0}/ProblemSetList.txt'.format(dataset_path))
    for problem_set in problem_sets:
        info[problem_set] = {}
        problems = read_legend('{0}/{1}/ProblemList.txt'.format(dataset_path, problem_set))
        for problem in problems:
            info[problem_set][problem] = {}
            mypath = '{0}/{1}/{2}/'.format(dataset_path, problem_set, problem)
            answer = read_legend('{0}/{1}/{2}/ProblemAnswer.txt'.format(dataset_path,problem_set, problem))[0]
            images = [file_name for file_name in listdir(mypath) if file_name[-3:] == 'png' and len(file_name) < 10]
            setup_length = len([file_name for file_name in images if file_name[0] < '9'])
            answers_length = len([file_name for file_name in images if file_name[0] > '9'])
            info[problem_set][problem]['setup_length'] = setup_length
            info[problem_set][problem]['answers_length'] = answers_length
            info[problem_set][problem]['answer'] = answer
    return info

def get_random_image_name_list(dataset_path):
    info = get_info(dataset_path)
    image_paths = []
    for problem_set in info.keys():
        for problem in info[problem_set].keys():
            for index in range(info[problem_set][problem]['setup_length'], - 1):
                file_name = str(index + 1)
                path = '{0}/{1}/{2}/{3}.png'.format(dataset_path, problem_set, problem, file_name)
                image_paths.append(path)

            for index in range(info[problem_set][problem]['answers_length'] - 1):
                file_name = chr(index + ord('A'))
                path = '{0}/{1}/{2}/{3}.png'.format(dataset_path, problem_set, problem, file_name)
                image_paths.append(path)
    random.shuffle(image_paths)
    return image_paths

def get_random_image_name(dataset_path):
    info = get_info(dataset_path)
    problem_set = random.choice(list(info.keys()))
    problem = random.choice(list(info[problem_set].keys()))
    # Toss a coin
    if random.randint(0, 1) > 0:
        # Select a setup image
        index = random.randint(0, info[problem_set][problem]['setup_length'] - 1)
        file_name = str(index + 1)
    else:
        # Select an answer image
        index = random.randint(0, info[problem_set][problem]['answers_length'] - 1)
        file_name = chr(index + ord('A'))
    path = '{0}/{1}/{2}/{3}.png'.format(dataset_path, problem_set, problem, file_name)
    return path

def get_image_batch(size, imsize=None, dataset_path):
    images = []
    image_names = get_random_image_name_list(dataset_path)
    if size > len(image_names):
        raise ValueError('Size is bigger than total image count.')
    for i in range(size):
        image = ndimage.imread(image_names[i], flatten=True)
        if imsize is not None:
            image = imresize(image, imsize)
        images.append(image)
    return np.array(images)

if __name__ == '__main__':
    print(get_image_batch(500, (28, 28), 'datasets/'))