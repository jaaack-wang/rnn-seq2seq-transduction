'''
Author: Zhengxiang (Jack) Wang 
GitHub: https://github.com/jaaack-wang
Website: https://jaaack-wang.eu.org
About: General utility functions.
'''
import os
from os import listdir, walk
from os.path import isfile, join, exists

import random
import json


def _create_dir(path):
    if not exists(path):
        os.mkdir(path)
        print(path + " created!")

        
def create_dir(path):
    pathes = path.split("/")
    if len(pathes) == 1:
        _create_dir(path)
    
    cur_p = "."
    for p in pathes:
        cur_p = join(cur_p, p)
        _create_dir(cur_p)


def read_data(filepath, skip=0, sep="\t"):
    data = []
    file = open(filepath, "r")
    
    for _ in range(skip):
        next(file)
    
    for line in file:
        line = line.strip("\n").split(sep)
        
        assert len(line) >= 2, "each line" \
        "must have two items separated by" \
        f"{sep} in {filepath}"
        
        data.append([line[0], line[-1]])

    return data


def save_ds_in_txt(ds, fp):
    tmp = "{}\t{}"
    with open(fp, "w") as f:
        f.write(tmp.format(ds[0][0], ds[0][1]))
        for d in ds[1:]:
            f.write("\n" + tmp.format(d[0], d[1]))
    
    f.close()
    print(fp + " saved!")


def read_json(fp):
    return json.load(open(fp, "r"))


def save_dict_as_json(dic, fp, indent=4):
    with open(fp, "w") as f:
        json.dump(dic, f, indent=indent)
        print(fp + " saved!")
        

def get_filepathes_from_dir(file_dir, include_sub_dir=False,
                            file_format=None, shuffle=False):
    
    if include_sub_dir:
        filepathes = []
        for root, _, files in walk(file_dir, topdown=False):
            for f in files:
                filepathes.append(join(root, f))
    else:
        filepathes = [join(file_dir, f) for f in listdir(file_dir)
                      if isfile(join(file_dir, f))]
        
    if file_format:
        if not isinstance(file_format, (str, list, tuple)):
            raise TypeError("file_format must be str, list or tuple.")
        file_format = tuple(file_format) if isinstance(file_format, list) else file_format
        format_checker = lambda f: f.endswith(file_format)
        filepathes = list(filter(format_checker, filepathes))

    if shuffle:
        random.shuffle(filepathes)
        
    return filepathes
