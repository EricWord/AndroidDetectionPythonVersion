# -*- coding:GBK -*-
import os
import os.path
import sys
import subprocess
import time

from featureProject import getFeatures

rootdir = "E:/7BiShe/badAPKs/3901-4200/3901-4200doing/"
destdir = "D:/cgs/File/data/0test0412/otherGoodApksRes/"
command = "java -jar D:/cgs/software/jar/apktool.jar"


class Packages:
    def __init__(self, srcdir, desdir):
        self.sdir = srcdir
        self.ddir = desdir

    def check(self):
        print("--------------------starting unpackage!---------------------")
        for dirpath, dirnames, filenames in os.walk(rootdir):
            for filename in filenames:
                thefile = os.path.join(dirpath, filename)
                apkfile = os.path.split(thefile)[1]
                apkname = os.path.splitext(apkfile)[0]
                print
                apkfile
                try:
                    if os.path.splitext(thefile)[1] == ".apk":
                        # name = os.path.splitext(thefile)[0]
                        str1 = '"' + thefile + '"'
                        str2 = '"' + destdir + os.path.splitext(filename)[0] + '"'
                        # cmdExtract = r'%s d -f %s %s'% (command, str2, str1)
                        getFeatures.main(thefile, apkname)
                        print
                        "******************well done******************"
                except IOError as err:
                    print(err)
                    sys.exit()


if __name__ == "__main__":
    dir = Packages(rootdir, 'd:/')
    dir.check()