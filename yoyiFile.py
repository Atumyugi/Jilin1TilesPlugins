# cython:language_level=3
# -*- coding: utf-8 -*-
import os
import os.path as osp
import traceback
import yaml
from shutil import copyfile
from osgeo import gdal
import numpy as np
import re
import xml.dom.minidom as xmldom
from xml.dom.minidom import Element

# 读取xml元数据文件
def readMetaXml(xmlFilePath):
    resDict = {}
    xmlF = xmldom.parse(xmlFilePath)
    eles : Element = xmlF.documentElement
    #print(eles.tagName,type(eles))
    productInfo : Element = eles.getElementsByTagName("ProductInfo")[0]
    sceneId = productInfo.getElementsByTagName("SceneID")[0].firstChild.data
    time = productInfo.getElementsByTagName("CenterTime")[0].firstChild.data
    #print(time,type(time))
    resDict["SceneID"] = sceneId
    resDict["CenterTime"] = time.split(" ")[0]
    return resDict
    #a = eles.getElementsByTagName("xmin")[0].firstChild.data


def checkTifList(tifFolder,onlyBaseName=False,sort=False):
    '''
    检查tiff文件列表 剔除不在json文件中tiff文件
    :param tiff_image_list: 待处理tiff文件夹
    :param json_path: 记录矩形框坐标信息的json文件
    :return: 待处理的tiff文件
    '''
    if onlyBaseName:
        if osp.isfile(tifFolder):
            return [osp.basename(tifFolder)]
        tiff_image_list = [i for i in os.listdir(tifFolder) if
                           i.split('.')[-1] in ['tif','TIF','TIFF']]
        # 对影像排序
        if sort:
            try:
                tiff_image_list.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
            except:
                pass
        return tiff_image_list
    else:
        if osp.isfile(tifFolder):
            return [tifFolder]
        tiff_image_list = [os.path.join(tifFolder, i) for i in os.listdir(tifFolder) if
                           i.split('.')[-1] in ['tif','TIF','TIFF']]
        if sort:
            try:
                tiff_image_list.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
            except:
                pass
        return tiff_image_list

def checkAllFileList(tifFolder,postType="tif"):
    postDict = {
        "tif" : ["tif","TIF","tiff","TIFF","GTIFF","gtiff"],
        "shp" : ["shp","SHP"]
    }
    tifList = []
    for root, dirs, files in os.walk(tifFolder):
        for file in files:
            if file.split(".")[-1] in postDict[postType]:
                tifList.append(osp.join(root,file))
    return tifList

# 检查shp文件列表 剔除不在json文件中tiff文件
def checkShpList(shpFolder):
    if osp.isfile(shpFolder):
        return [shpFolder]
    shpFolderList = [os.path.join(shpFolder, i) for i in os.listdir(shpFolder) if
                       i.split('.')[-1] in ['shp']]
    return shpFolderList

# 检查png文件列表 剔除不在json文件中png,jpg文件
def checkImgList(imgFolder):
    if osp.isfile(imgFolder):
        return [imgFolder]
    FolderList = [os.path.join(imgFolder, i) for i in os.listdir(imgFolder) if
                       i.split('.')[-1] in ['png','jpg']]
    return FolderList

def checkPyAndPth(yamlFolder):
    pth = [os.path.join(yamlFolder, i) for i in os.listdir(yamlFolder) if i.split('.')[-1] in ['pth','dll']][0]
    py = [os.path.join(yamlFolder, i) for i in os.listdir(yamlFolder) if i.split('.')[-1] in ['py']][0]
    return py,pth

def checkPyAndPthAndYamlComplete(modelDir):
    pth = osp.join(modelDir,"cgioiy.dll")
    py = osp.join(modelDir,"cgconfig.py")
    yaml = osp.join(modelDir,"yaoi")
    if osp.exists(pth) and osp.exists(py) and osp.exists(yaml):
        return True
    else:
        return False


# 判断影像文件夹的波段数量，如果都一样则返回波段数量，如果不一样返回-1
def checkTifDirHaveSameBand(tifDir):
    tifList = checkTifList(tifDir)
    if len(tifList) > 0:
        resBand = -1
        for tif in tifList:
            tempDs: gdal.Dataset = gdal.Open(tif)
            if resBand==-1:
                resBand = tempDs.RasterCount
            elif resBand != tempDs.RasterCount:
                return -1
        return resBand
    else:
        return -1

# 批量过滤尺寸大小不对的tif
def clearTifForSize(tifBaseNames,dir,size=512):
    res = []
    for tif in tifBaseNames:
        tifDs : gdal.Dataset = gdal.Open(osp.join(dir,tif))
        if min(tifDs.RasterXSize,tifDs.RasterYSize) >= size:
            res.append(tif)
    # 对影像排序
    res.sort(key = lambda i:int(re.match(r'(\d+)',i).group()))
    return res

def clearTifForNoValid(tifBaseNames,imgDir):
    for tif in tifBaseNames:
        tifDs : gdal.Dataset = gdal.Open(osp.join(imgDir,tif))
        tifNp = tifDs.ReadAsArray()[0]
        if np.max(tifNp) == tifDs.GetRasterBand(1).GetNoDataValue():
            del tifNp
            del tifDs
            os.remove(osp.join(imgDir,tif))
            print("删除了",tif)


# 检查影像和矢量是不是一对一存在
def checkTifShpPair(tifDir,shpDir):
    tifList = checkTifList(tifDir)
    for tifPath in tifList:
        tempShpName = osp.basename(tifPath).split(".")[0] + ".shp"
        shpPath = osp.join(shpDir, tempShpName)
        if not osp.exists(shpPath):
            return False
    return True

# 检查MMSEG数据集的完整性 complete 完整   其他 各种错误提示
def checkMMSegDataSet(dataSetDir) -> str:
    imgDir = osp.join(dataSetDir,"img")
    maskDir = osp.join(dataSetDir,"mask")
    yaml = osp.join(dataSetDir,"dataSetSetting.yaml")
    trainTxt = osp.join(dataSetDir,"train.txt")
    validTxt = osp.join(dataSetDir, "valid.txt")
    if not osp.exists(imgDir):
        return "不存在img文件夹"
    if not osp.exists(maskDir):
        return "不存在mask文件夹"
    if not osp.exists(yaml):
        return "不存在dataSetSetting.yaml文件"
    if not osp.exists(trainTxt):
        return "不存在train.txt文件"
    if not osp.exists(validTxt):
        return "不存在valid.txt文件"
    return "complete"

# 检查文件夹中的一级子文件夹
def checkChildDirI(dir):
    for root,dirs,files in os.walk(dir):
        return dirs

## ———————————— os 改变文件相关 so ———————————— ##

# 删除文件夹
def deleteDir(dir,delRoot=True):
    for root, dirs, files in os.walk(dir):
        for file in files:
            try:
                os.remove(os.path.join(root, file))
            except:
                pass
        for dir in dirs:
            try:
                os.rmdir(os.path.join(root, dir))
            except:
                pass
        if delRoot:
            try:
                os.rmdir(root)
            except:
                pass

# 删除矢量
def deleteShp(shpPath):
    shpDir = osp.dirname(shpPath)
    fnameNoExt = osp.basename(shpPath).split(".")[0]
    extensions = ["shp","shx","dbf","prj","sbn","sbx","fbn","fbx","ain","aih","ixs","mxs","atx","xml","cpg","qix","ysi"]
    toDelFiles = [ osp.join(shpDir,fnameNoExt+f".{ext}") for ext in extensions]
    try:
        for f in toDelFiles:
            if osp.exists(f):
                os.remove(f)
    except:
        print("删除失败，请解除占用")
# 创建文件夹
def createDir(dir):
    if not osp.exists(dir):
        os.mkdir(dir)

# 复制文件
def copyFile(path,outPath):
    try:
        copyfile(path,outPath)
    except:
        print("复制失败")

# 将影像文件夹分配到N个子文件夹中
def assignTifDir(tifDir,outDir,personNum,callback=None):
    baseName = osp.basename(tifDir)
    tifList = checkTifList(tifDir)
    assignTifNum = len(tifList) // personNum
    tifIndex = 0
    for i in range(personNum):
        assignTempDir = osp.join(outDir,f"{baseName}_{i+1}")
        createDir(assignTempDir)
        for i in range(assignTifNum):
            if tifIndex < len(tifList):
                if callback:
                    callback( ((tifIndex+1)/len(tifList)) * 100 )
                copyfile(tifList[tifIndex],osp.join(assignTempDir,osp.basename(tifList[tifIndex])))
                tifIndex += 1
            else:
                break
    while tifIndex < len(tifList):
        copyfile(tifList[tifIndex], osp.join(assignTempDir, osp.basename(tifList[tifIndex])))
        tifIndex += 1

# 将两个变化检测的影像文件数量变为一样的
def fitPreLateTifDir(preDir,lateDir):
    preList = checkTifList(preDir,True)
    lateList = checkTifList(lateDir,True)
    if len(preList) > len(lateList):
        for tifName in preList:
            if not osp.exists(osp.join(lateDir,tifName)):
                os.remove(osp.join(preDir,tifName))
    elif len(preList) < len(lateList):
        for tifName in lateDir:
            if not osp.exists(osp.join(preDir,tifName)):
                os.remove(osp.join(lateDir,tifName))

# 剔除掉两个列表中名字不一样的影像地址,并最后返回一个一对一的字典
def fitPreLateListToDict(preList,lateList):
    resDict = {}
    preBaseList = []
    for prePath in preList:
        preBaseList.append(osp.basename(prePath))
    lateBaseList = []
    for latePath in lateList:
        lateBaseList.append(osp.basename(latePath))

    for prePath,i in zip(preBaseList,range(len(preBaseList))):
        if prePath in lateBaseList:
            lateIndex = lateBaseList.index(prePath)
            resDict[preList[i]] = lateList[lateIndex]

    return resDict


# 计算文件大小
def getFileSize(filePath):
    fsize = osp.getsize(filePath)
    if fsize < 1024:
        return f"{round(fsize,2)} Byte"
    else:
        KBX = fsize/1024
        if KBX < 1024:
            return f"{round(KBX,2)} Kb"
        else:
            MBX = KBX / 1024
            if MBX < 1024:
                return f"{round(MBX,2)} Mb"
            else:
                GBX = MBX / 1024
                return f"{round(GBX,2)} Gb"

# 创建简单的yaml文件，属性用字典表示
def createTxtForDict(txtPath,contentDict:dict):
    with open(txtPath, 'w') as f:
        for key,value in contentDict.items():
            f.write(f"{key},{value}")

def saveYamlForDict(savePath,contentDict:dict):
    with open(savePath,"w",encoding="utf-8") as f:
        yaml.dump(contentDict,f,allow_unicode=True)

# 保存样本辅助勾画的yaml文件,cateName="",catePixelValue=1,cateCode="1"
def saveSampleWorkYaml(savePath:str,workTypeMode:int,preLayer=None,lateLayer=None,curIndex=0,segSize=None,cateName="",catePixelValue=1,cateCode="1"):
    workTypeDict = {
        1 : "segment",
        2 : "changeDetec",
        3 : "wmsSegment",
        4 : "wmsChangeDetec",
        5 : "objectDetec"
    }
    contentDict = {
        "workType": workTypeDict[workTypeMode],
        "preLayer": preLayer,
        "lateLayer": lateLayer,
        "curIndex": curIndex,
        "segSize" : segSize,
        "cateName" : cateName,
        "catePixelValue" : catePixelValue,
        "cateCode": cateCode
    }
    saveYamlForDict(savePath,contentDict)

def readYamlToDict(yamlPath)->dict:
    try:
        with open(yamlPath, 'rb') as stream:
            dict = yaml.load(stream, Loader=yaml.FullLoader)
        return dict
    except:
        print(traceback.format_exc())
        return {}

if __name__ == '__main__':
    cfg = r"C:\qgis322\samDrawer\#workDir\water_001\workDir.cfg"
    saveYamlForDict(cfg,{"curIndex":0})
