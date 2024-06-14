# coding: utf-8

import sys
import os
import re

dir = os.path.split(os.path.abspath(__file__))[0]
sh = dir + "/hsiclasso.sh"

my_f_num = input('Number of features to select : ')
my_learn = input('r(regression) or c(classification)? : ')
my_B = input('B : ')
my_neig = input('Number of neighbors : ')

s = f'python {dir}/myHSICLasso.py {dir}/all_path.txt '

if my_f_num:
    s = s + f'-f {my_f_num} '

if my_learn:
    s = s + f'-l {my_learn} '

if my_B:
    s = s + f'-b {my_B} '
    
if my_neig:
    s = s + f'-n {my_neig}'
    
with open(sh,'w') as f:
    f.write(s)

print(f'Finish making a shell script file\n    FileName : {sh}\n    Caution : Above script execute feature selction with HSIClasso')
