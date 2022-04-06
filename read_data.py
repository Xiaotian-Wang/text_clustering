import json
import re

with open('HKBK_CT.json', encoding='utf-8') as f:
    data = json.load(f)

with open('HKBK_GXB_FL_CT.json', encoding='utf-8') as f:
    datawithcate = json.load(f)

a = [{'name': item['SYZWMC'], 'text':item['YSCTSW'], 'doc':item['WJMC']} for item in data['RECORDS'] if item['WJMC']=='空军航空工程词典']
b = [{'name': item['SYZWMC'], 'cate':item['FLMC'], 'code':item['FLDM']} for item in datawithcate['RECORDS']]

anames = [item['name'] for item in a]
bnames = [item['name'] for item in b]

dr = re.compile(r'<[^>]+>', re.S)

intersection = [item for item in a if item['name'] in bnames]
for i in range(len(intersection)):
    item = intersection[i]
    index_in_b = bnames.index(item['name'])
    cate = b[index_in_b]['cate']
    code = b[index_in_b]['code']
    intersection[i]['cate'] = cate
    intersection[i]['code'] = code
    intersection[i]['text'] = dr.sub('', intersection[i]['text'])


intersection = [item for item in intersection if len(item['code']) == 4]

with open('dataset.json', 'w', encoding='utf-8') as f:
    json.dump(intersection, f, ensure_ascii=False, indent=2)