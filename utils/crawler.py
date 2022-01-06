# -*- coding:UTF-8 -*-
"""
@author: dueToLife
@contact: wkp372874136@mail.ustc.edu.cn
@datetime: 2021/12/25 16:06
@file: crawler.py
@software: PyCharm
"""
import requests
import re

if __name__ == "__main__":
    pattern = re.compile(r'href="([0-9]+)(_)(.*)(_)(now|then)(\.)(jpg|png)"')
    print(pattern)
    session = requests.Session()
    url = "https://www.fourmilab.ch/images/lignieres_then_and_now/figures/"
    text = session.get(url=url)
    res = pattern.findall(text.text)
    print(len(res))
    for i in range(len(res)):
        file_name = ""
        local_name = ""
        for j in range(len(res[i])):
            file_name += res[i][j]
            if j != 2 and j != 1:
                local_name += res[i][j]
        print(file_name, local_name)
        pic = session.get(url=url+file_name)
        with open('../figs/'+local_name, 'wb') as file:
            file.write(pic.content)
            file.close()
