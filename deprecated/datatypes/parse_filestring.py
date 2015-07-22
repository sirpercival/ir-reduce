# -*- coding: utf-8 -*-
"""
Created on Thu May 28 02:35:18 2015

@author: gray
"""

import re
placeholder = '#'
reg = placeholder+'+'
from stacks import ImageStack

def parse_filestring(filestring, stub):
    if len(re.findall(reg, stub)) != 1:
        raise ValueError("File format is not valid; must use '#' as placeholder only")
    spot = re.search(reg, stub)
    spotlen = str(spot.end() - spot.start())
    base = re.sub(reg, '%0'+spotlen+'d', stub)
    
    files = []
    tmp = re.split('[.,]', filestring)
    for t in tmp:
        f = re.split('-', t)
        if len(f) == 1:
            files.append(int(f))
        else:
            for i in range(len(f) - 1):
                for j in range(int(f[i]), int(f[i+1])+1):
                    files.append(j)
    images = ImageStack([base + '.fits' % x for x in files], dither=True)
    return images
