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


def get_permissions(src):
    app = apk.APK(src)
    permission = app.get_permissions()
    file = permission
    perms=extractAuthority(file)
    return perms




if __name__ == '__main__':
    # src = "E:/BiSheData/goodApks/com.hudongzuoye.apk"
    # print(get_permissions(src))

    a = []
    for i in range(1, len(sys.argv)):
        a.append((sys.argv[i]))

    print(get_permissions(a[0]))