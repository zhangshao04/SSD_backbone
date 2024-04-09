import xml.etree.ElementTree as ET
import os

def move_size_before_objects(xml_file):
    # 解析XML文件
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size_element = root.find('size')
    if size_element is not None:
        root.remove(size_element)

    object_index = next((i for i, elem in enumerate(root) if elem.tag == 'object'), None)
    
 
    if object_index is not None:
        root.insert(object_index, size_element)
    else:
        root.append(size_element)

    tree.write(xml_file, encoding='utf-8', xml_declaration=True)


dir_path = r'C:\users\zhangyf\SSD\VOC_to_COCO\VOC2007_test'
for f in os.listdir(dir_path):
    xml_file = os.path.join(dir_path,f)
    move_size_before_objects(xml_file)

