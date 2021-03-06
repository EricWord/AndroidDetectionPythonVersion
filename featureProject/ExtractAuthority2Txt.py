# coding=utf-8

from androguard.core.bytecodes import apk, dvm
import sys
'''
提取权限特征
'''
def extractAuthority(file):
    authorities=[]
    for i in file:
        tmp = i.split('.')
        final = tmp[-1]
        authorities.append(final)
    return  authorities


def writeToTxt(file):
    fm = open('E:/BiSheData/temp/res' + '.txt', 'w')
    for i in file:
        tmp = i.split('.')
        final = tmp[-1]
        fm.write(final)
        fm.write("\n")
    fm.close()

def get_permissions(src):
    app = apk.APK(src)
    permission = app.get_permissions()
    file = permission
    perms=extractAuthority(file)
    writeToTxt(file)
    return perms




if __name__ == '__main__':

    a = []
    for i in range(1, len(sys.argv)):
        a.append((sys.argv[i]))

    print(get_permissions(a[0]))