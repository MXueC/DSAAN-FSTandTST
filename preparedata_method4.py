
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def diff_img(imgs_path_lt,start_index,stop_index):
     """function: generate a gray diff map according to imgs seq
     imgs_path_lt:path list
     start_index:start index
     stop_index:stop index"""
     if stop_index<start_index:
         print("index erro,stopindex<startindex")
         return None
     if not len(imgs_path_lt):
         print("empty list")
         return None
     if start_index<0:
         start_index=0
     if stop_index>len(imgs_path_lt)-1:
         stop_index= len(imgs_path_lt)-1
     if start_index==stop_index:
         cur_img = cv2.imread(imgs_path_lt[start_index])
         cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
         return cur_img


     # # method c
     # #
     cur_img = cv2.imread(imgs_path_lt[start_index])
     next_img = cv2.imread(imgs_path_lt[stop_index])
     cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
     next_img = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)
     difmap = cv2.absdiff(cur_img, next_img)
     return difmap


if __name__ == "__main__":

     pass