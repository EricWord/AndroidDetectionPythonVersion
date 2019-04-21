#!/usr/bin/env python
# coding=utf-8
"""
@file: mi_shop.py
@contact: bianzhiwei@iyoujia.com
@time   : 2019/2/27 13:46 
@Desc   :
"""

# coding=utf-8
import queue
import threading
import urllib
import requests
import re
from bs4 import BeautifulSoup

# 用对存储任务
url_queue = queue.Queue()


def parser_apks(page_num=1):
    _root_url = "http://app.mi.com"  # 应用市场主页网址
    res_parser = {}
    # 设置爬取的页面，从第一页开始爬取，第一页爬完爬取第二页，以此类推

    # 获取应用列表页面
    wbdata = requests.get("http://app.mi.com/catTopList/10?page=" + str(page_num)).text
    print("开始爬取第" + str(page_num) + "页")
    # 解析应用列表页面内容
    soup = BeautifulSoup(wbdata, "html.parser")
    links = soup.find_all("a", href=re.compile("/details?"), class_="", alt="")
    for link in links:
        # 获取应用详情页面的链接
        detail_link = urllib.parse.urljoin(_root_url, str(link["href"]))
        package_name = detail_link.split("=")[1]
        download_page = requests.get(detail_link).text
        # 解析应用详情页面
        soup1 = BeautifulSoup(download_page, "html.parser")
        download_link = soup1.find(class_="download")["href"]
        # 获取直接下载的链接
        download_url = urllib.parse.urljoin(_root_url, str(download_link))
        # 解析后会有重复的结果，通过判断去重
        res_parser[download_url] = package_name

    # print("爬取apk数量为: " + str(len(res_parser)))
    res_parser = {value: key for key, value in res_parser.items()}
    print("爬取apk数量为: " + str(len(res_parser)))
    return res_parser


def down_apk():
    while True:
        try:
            args = url_queue.get()
            urllib.request.urlretrieve(args[0], args[1])
            print('下载应用 %s 完成' % args[1].split('\\')[-1])
        except Exception:
            print('queue 中任务已经完成  没有新任务')


def craw_apks(save_path="E:\\office1\\"):
    """
    真正耗费时间的是下载apk  由网速决定  所以只给下载apk的方法 使用多线程
    :param save_path:
    :return:
    """
    # 设置 10个线程
    for i in range(10):
        thread = threading.Thread(target=down_apk)
        thread.start()
        thread.join(0.1)
    for page_num in range(24, 56):
        # 进行翻页
        res_dic = parser_apks(page_num)
        for apk in res_dic.keys():
            apk_url = res_dic[apk]
            path = save_path + apk + ".apk"
            # 将获取的url  和存储路径放进任务队列中
            url_queue.put([apk_url, path])


if __name__ == "__main__":
    craw_apks()
