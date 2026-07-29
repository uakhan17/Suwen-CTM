from collections import defaultdict
from pathlib import Path
import os, re, warnings
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib.font_manager as fm

# Suppress glyph warnings
warnings.filterwarnings("ignore", message="Glyph .* missing from font.*")

# Setup font
FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fm.fontManager.addfont(FONT_PATH)
font_prop = fm.FontProperties(fname=FONT_PATH)
FONT_NAME = font_prop.get_name()

import matplotlib as mpl
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': [FONT_NAME],
    'axes.unicode_minus': False,
})

# 全标签定义
ALL_LABELS = {
    '舌质_神'   : ['枯舌', '荣舌'],
    '舌质_色'   : ['淡红', '红', '淡白', '绛红', '青紫'],
    '舌质_形'   : ['老', '嫩', '胖', '瘦', '齿痕', '点刺', '裂纹'],
    '舌质_态'   : ['痿软', '强硬', '歪斜', '颤动', '吐弄', '短缩'],
    '舌苔_苔色' : ['灰黑', '白', '黄', '无', '滑', '少'],
    '舌苔_苔质' : ['燥', '薄', '厚', '润', '腻', '腐', '剥落', '偏全'],
    '望眼_目色' : ['目胞色黑晦暗', '白睛发黄', '两眦淡白', '目赤肿痛'],
    '望眼_目态' : ['目睛凝视', '胞睑下垂', '睡眠露睛', '目睛上视', '斜视'],
    '望眼_目形' : ['眼球凹陷', '眼球突出', '胞睑红肿', '目胞浮肿', '无异常'],
    '望眼_瞳孔' : ['瞳孔缩小', '瞳孔等大', '瞳孔散大'],
    '望口_口形' : ['口角无异常', '口歪不收', '口疮', '口糜', '鹅口疮'],
    '望口_口态' : ['口张', '口噤', '口撮', '口', '口振', '口动', '无异常'],
    '望唇_唇色' : ['淡白', '深红', '赤红', '樱桃红', '青紫', '青黑', '红润'],
    '望唇_唇形' : ['无异常', '唇干而裂', '嘴唇糜烂', '唇内溃烂', '唇边生疮', '唇角生疔', '口唇翻卷不能覆齿'],
    '望鼻_鼻色' : ['色白', '色赤', '微黄', '灰暗枯槁', '红黄隐隐 明润含蓄'],
    '望鼻_鼻形' : ['鼻头生疖', '生粉刺', '鼻翼扇动', '鼻柱溃陷', '无异常'],
    '面色_面色'  : ['青黄','红赤满面通红','午后潮红','久病苍白却时颧赤泛红如妆','淡白无华','晄白','苍白','淡青','青黑','青灰','青黄','黧黑晦暗','紫暗黧黑','黑而干焦','眼眶发黑','萎黄','黄胖','阳黄','阴黄','面黄','红黄隐隐明润含蓄'],
    '面色_皮肤光泽': ['荣润','枯槁'],
    '面色_面形'   : ['无异常','面肿','腮肿','面削颧耸','口眼斜','惊恐貌','苦笑貌'],
}

def _plot_series(s: pd.Series, title: str, out_path: Path) -> None:
    """绘制彩色柱状图并保存到 out_path"""
    # DEBUG: Ensure all categories are present
    print(f"[DEBUG] Plotting: {title}")
    print(f"[DEBUG] Labels: {list(s.index)}")

    cmap = colormaps.get_cmap('tab20').resampled(len(s))
    colors = [cmap(i) for i in range(len(s))]

    ax = s.plot(kind='bar',
                color=colors,
                width=0.8,
                figsize=(8, 4))

    ax.set_title(title, fontproperties=font_prop)
    ax.set_xlabel('')
    ax.set_ylabel('计数', fontproperties=font_prop)

    plt.xticks(rotation=45, ha='right', fontproperties=font_prop)
    plt.yticks(fontproperties=font_prop)

    # Annotate counts on top of bars
    for i, (label, count) in enumerate(s.items()):
        ax.text(i, count + 0.5, str(count), ha='center', va='bottom',
                fontproperties=font_prop, fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()

def _parse_csv(path: Path) -> pd.DataFrame:
    """读取单个 csv，把第 1 行作为子标签合并进列名后返回 DataFrame"""
    df = pd.read_csv(path)
    sub_headers = df.iloc[0]
    new_cols, current_parent = [], None

    for col, sub in zip(df.columns, sub_headers):
        col_clean = str(col).strip()
        if not col_clean.startswith('Unnamed'):
            current_parent = col_clean

        if col_clean == '图片名':
            new_cols.append('图片名')
            continue

        sub_clean = str(sub).strip()
        new_cols.append(f'{current_parent}_{sub_clean}')

    df.columns = new_cols
    return df.iloc[1:].reset_index(drop=True)

def _label_distribution(df: pd.DataFrame) -> dict[str, pd.Series]:
    out = {}
    for col in df.columns:
        if col == '图片名':
            continue

        exploded = (
            df[col]
            .dropna()
            .astype(str)
            .str.replace('，', ',', regex=False)
            .str.split(r'\s*,\s*')
            .explode()
        )
        counts = exploded.value_counts()

        if col in ALL_LABELS:
            counts = counts.reindex(ALL_LABELS[col]).fillna(0).astype(int)

        out[col] = counts
    return out

def main(data_dir='.', pattern='*.csv', plot=True):
    files = sorted(Path(data_dir).glob(pattern))
    if not files:
        print(f'No CSV files found in {data_dir}')
        return

    for f in files:
        print(f'\n========== {f.name} ==========')
        df = _parse_csv(f)
        dists = _label_distribution(df)

        for col, counts in dists.items():
            print(f'\n[{col}]')
            print(counts.to_string())
            if True:
                safe = re.sub(r'[^\w\-_一-龥]', '_', col)
                _plot_series(counts, f'{f.stem} | {col}', Path('plots')/f'{f.stem}_{safe}.png')

if __name__ == '__main__':
    import argparse, textwrap
    parser = argparse.ArgumentParser(
        description='统计面诊/舌诊 CSV 标签分布',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''
            示例:
                python label_stats.py -d /mnt/data --plot
        '''))
    parser.add_argument('-d', '--data_dir', default='.', help='CSV 文件夹')
    parser.add_argument('-p', '--pattern', default='*.csv', help='文件匹配模式')
    parser.add_argument('--plot', action='store_true', help='是否画柱状图')
    args = parser.parse_args()
    main(args.data_dir, args.pattern, args.plot)
