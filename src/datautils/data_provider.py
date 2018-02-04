from os import listdir
import random
import scipy.ndimage as ndimage
from scipy.misc import imresize

def read_legend(legend_name):
    with open(legend_name) as file:
        filename_list = [x.strip() for x in file.readlines()]
    return filename_list

def get_info():
    info = {}
    problem_sets = read_legend('datasets/ProblemSetList.txt')
    for problem_set in problem_sets:
        info[problem_set] = {}
        problems = read_legend('datasets/{0}/ProblemList.txt'.format(problem_set))
        for problem in problems:
            info[problem_set][problem] = {}
            mypath = 'datasets/{0}/{1}/'.format(problem_set, problem)
            answer = read_legend('datasets/{0}/{1}/ProblemAnswer.txt'.format(problem_set, problem))[0]
            images = [file_name for file_name in listdir(mypath) if file_name[-3:] == 'png' and len(file_name) < 10]
            setup_length = len([file_name for file_name in images if file_name[0] < '9'])
            answers_length = len([file_name for file_name in images if file_name[0] > '9'])
            info[problem_set][problem]['setup_length'] = setup_length
            info[problem_set][problem]['answers_length'] = answers_length
            info[problem_set][problem]['answer'] = answer
    return info

def get_random_image_name():
    info = get_info()
    problem_set = random.choice(list(info.keys()))
    problem = random.choice(list(info[problem_set].keys()))
    # Toss a coin
    if random.randint(0, 1) > 0:
        # Select a setup image
        file_name = random.randint(1, info[problem_set][problem]['setup_length'])
    else:
        # Select an answer image
        index = random.randint(0, info[problem_set][problem]['answers_length'] - 1)
        file_name = chr(index + ord('A'))
    path = 'datasets/{0}/{1}/{2}.png'.format(problem_set, problem, file_name)
    return path

def get_image_batch(size, imsize=None):
    images = []
    for i in range(size):
        path = get_random_image_name()
        image = ndimage.imread(path, flatten=True)
        if imsize is not None:
            image = imresize(image, imsize)
        images.append(image)
    return images

if __name__ == '__main__':
    print(get_image_batch(1000))
        