# build_label.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv, argparse, pathlib, json

ALL_LABELS = {
    '舌质_神': ['枯舌', '荣舌'],
    '舌质_色': ['淡红', '红', '淡白', '绛红', '青紫'],
    '舌质_形': ['老', '嫩', '胖', '瘦', '齿痕', '点刺', '裂纹'],
    '舌质_态': ['痿软', '强硬', '歪斜', '颤动', '吐弄', '短缩'],
    '舌苔_苔色': ['灰黑', '白', '黄', '无', '滑', '少'],
    '舌苔_苔质': ['燥', '薄', '厚', '润', '腻', '腐', '剥落', '偏全'],
    '望眼_目色': ['目胞色黑晦暗', '白睛发黄', '两眦淡白', '目赤肿痛'],
    '望眼_目态': ['目睛凝视', '胞睑下垂', '睡眠露睛', '目睛上视', '斜视'],
    '望眼_目形': ['眼球凹陷', '眼球突出', '胞睑红肿', '目胞浮肿', '无异常'],
    '望眼_瞳孔': ['瞳孔缩小', '瞳孔等大', '瞳孔散大'],
    '望口_口形': ['口角无异常', '口歪不收', '口疮', '口糜', '鹅口疮'],
    '望口_口态': ['口张', '口噤', '口撮', '口', '口振', '口动', '无异常'],
    '望唇_唇色': ['淡白', '深红', '赤红', '樱桃红', '青紫', '青黑', '红润'],
    '望唇_唇形': ['无异常', '唇干而裂', '嘴唇糜烂', '唇内溃烂', '唇边生疮', '唇角生疔', '口唇翻卷不能覆齿'],
    '望鼻_鼻色': ['色白', '色赤', '微黄', '灰暗枯槁', '红黄隐隐 明润含蓄'],
    '望鼻_鼻形': ['鼻头生疖', '生粉刺', '鼻翼扇动', '鼻柱溃陷', '无异常'],
    '面色_面色': ['青黄','红赤满面通红','午后潮红','久病苍白却时颧赤泛红如妆','淡白无华','晄白',
                '苍白','淡青','青黑','青灰','青黄','黧黑晦暗','紫暗黧黑','黑而干焦','眼眶发黑',
                '萎黄','黄胖','阳黄','阴黄','面黄','红黄隐隐明润含蓄'],
    '面色_皮肤光泽': ['荣润','枯槁'],
    '面色_面形': ['无异常','面肿','腮肿','面削颧耸','口眼斜','惊恐貌','苦笑貌'],
}

FACE_TASKS   = [t for t in ALL_LABELS if t.startswith('望') or t.startswith('面色')]
TONGUE_TASKS = [t for t in ALL_LABELS if t.startswith('舌')]

parser = argparse.ArgumentParser()
parser.add_argument('--group', choices=['face','tongue','all'], default='all')
parser.add_argument('--json', help='外部 JSON 标签表')
parser.add_argument('--out', help='输出文件名')
args = parser.parse_args()

# 若提供 JSON，则覆盖 ALL_LABELS
if args.json:
    with open(args.json, 'r', encoding='utf-8') as f:
        ALL_LABELS = json.load(f)
    FACE_TASKS   = [t for t in ALL_LABELS if t.startswith('望') or t.startswith('面色')]
    TONGUE_TASKS = [t for t in ALL_LABELS if t.startswith('舌')]

if args.group == 'face':
    selected = {k:v for k,v in ALL_LABELS.items() if k in FACE_TASKS}
elif args.group == 'tongue':
    selected = {k:v for k,v in ALL_LABELS.items() if k in TONGUE_TASKS}
else:
    selected = ALL_LABELS

out_path = pathlib.Path(args.out or f'label_map_{args.group}.csv')
with out_path.open('w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['task', 'label', 'index'])
    for task, labels in selected.items():
        for idx, lab in enumerate(labels):
            writer.writerow([task, lab, idx])

print(f'✔️  {out_path} 生成完毕，共 {sum(len(v) for v in selected.values())} 条映射')
