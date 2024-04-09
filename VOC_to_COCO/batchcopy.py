import os
import shutil


# the path is you original file directory
# the newpath is the new directory
class BatchCopy():
    def __init__(self):
        self.path = r'C:\users\zhangyf\SSD\ssd\data\datasets\VOC07+12+test\VOCdevkit\VOC2007\Annotations'  ####voc是将所有xml文件都放在同一目录下
        self.image_path = r'C:\users\zhangyf\SSD\ssd\data\datasets\VOC07+12+test\VOCdevkit\VOC2007\JPEGImages'####voc的图片位置
        self.newpath = r'C:\users\zhangyf\SSD\VOC_to_COCO\VOC2007_test'  ####将训练集的xml文件单独放一个目录
        self.newiamge_path = r'C:\users\zhangyf\SSD\VOC_to_COCO\VOC2007_test_JPG'####voc转coco后的图片分类位置
        self.txt = r'C:\users\zhangyf\SSD\ssd\data\datasets\VOC07+12+test\VOCdevkit\VOC2007\ImageSets\Main\test.txt' ###训练集或验证集的txt文件位置

    def copy_file(self):
        if not os.path.exists(self.newpath):
            os.makedirs(self.newpath)
        else:
            shutil.rmtree(self.newpath)
        if not os.path.exists(self.newiamge_path):
            os.makedirs(self.newiamge_path)



        filelist = os.listdir(self.path)  # file list in this directory
        # print(len(filelist))
        test_list = loadFileList(self.txt)
        # print(len(test_list))
        for f in filelist:
            filedir = os.path.join(self.path, f)
            (shotname, extension) = os.path.splitext(f)
            if str(shotname) in test_list:
                filedir_image = os.path.join(self.image_path, shotname) + '.jpg'
                # print('success')
                shutil.copyfile(str(filedir_image),os.path.join(self.newiamge_path,shotname)+'.jpg')
                shutil.copyfile(str(filedir), os.path.join(self.newpath, f))


# load the list of train/test file list
def loadFileList(txt):
    filelist = []
    f = open(txt)  
    lines = f.readlines()
    for line in lines:
        line = line.strip('\r\n')  # to remove the '\n' for test.txt, '\r\n' for tainval.txt
        line = str(line)
        line = os.path.basename(line)

        _, file_suffix = os.path.splitext(line)
        filelist.append(_)

    f.close()
    # print(filelist)
    return filelist


if __name__ == '__main__':
    demo = BatchCopy()
    demo.copy_file()
