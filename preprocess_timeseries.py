import pandas as pd
import argparse
import os

def merge_datetime(input_path, output_path=None, drop_original=False):
    if input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    elif input_path.endswith('.xls') or input_path.endswith('.xlsx'):
        df = pd.read_excel(input_path)
    else:
        raise ValueError('仅支持CSV或Excel文件！')

    if all(col in df.columns for col in ['year', 'month', 'day', 'hour']):
        df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        print('已合成 datetime 列！')
        if drop_original:
            df = df.drop(columns=['year', 'month', 'day', 'hour'])
            print('已删除 year, month, day, hour 原始列！')
    else:
        print('未检测到 year, month, day, hour 四列，未做处理。')

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = base + '_datetime.csv'
    df.to_csv(output_path, index=False)
    print(f'处理后数据已保存为: {output_path}')
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='合成 year,month,day,hour 为 datetime 列')
    parser.add_argument('input_path', help='输入文件路径（csv/xls/xlsx）')
    parser.add_argument('--output', help='输出文件路径（默认自动命名）', default=None)
    parser.add_argument('--drop', action='store_true', help='是否删除原始 year,month,day,hour 列')
    args = parser.parse_args()
    merge_datetime(args.input_path, args.output, args.drop)