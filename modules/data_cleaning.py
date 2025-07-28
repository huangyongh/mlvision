import streamlit as st
import pandas as pd
import numpy as np

def run():
    st.header("🧹 数据清洗模块（时序友好）")

    file = st.file_uploader("上传数据文件（支持 CSV, Excel）", type=["csv", "xls", "xlsx"], key="data_cleaning_upload")
    df = None

    if file is not None:
        # 只在新文件上传时处理
        if "last_uploaded_file" not in st.session_state or st.session_state["last_uploaded_file"] != file.name:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.name.endswith('.xls') or file.name.endswith('.xlsx'):
                df = pd.read_excel(file)
            else:
                st.error("仅支持CSV或Excel文件！")
                st.stop()
            # 清空所有相关 session_state
            for key in [
                "outlier_last_affected", "outlier_show_flag",
                "fill_last_affected", "fill_last_deleted",
                "del_cols", "proc_cols", "outlier_cols", "fill_cols", "del_na_cols"
            ]:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.data_cleaning_df = df.copy()
            st.session_state._last_file_name = file.name
            st.session_state["last_uploaded_file"] = file.name
            st.experimental_rerun() if hasattr(st, 'experimental_rerun') else st.rerun()
        else:
            df = st.session_state.data_cleaning_df
    elif "data_cleaning_df" in st.session_state:
        df = st.session_state.data_cleaning_df
    else:
        st.info("请先上传数据文件")
        st.stop()

    # 删除不需要的列
    st.subheader("🗑️ 删除不需要的列")
    with st.form("del_cols_form"):
        del_cols = st.multiselect("选择要删除的列（可多选）", df.columns.tolist(), key="del_cols")
        submitted = st.form_submit_button("删除所选列")
        if submitted and del_cols:
            df = df.drop(columns=del_cols)
            st.session_state.data_cleaning_df = df
            st.info(f"已删除列：{del_cols}")
            st.experimental_rerun() if hasattr(st, 'experimental_rerun') else st.rerun()
    if len(df.columns) == 0:
        st.warning("所有列都被删除，请至少保留一列！")
        st.stop()

    # 数据概览
    st.subheader("🔍 数据概览")
    time_candidates = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])]
    sorted_df = df
    sorted_time_col = None
    for col in time_candidates:
        try:
            try:
                temp = pd.to_datetime(df[col], format="%Y-%m-%d", errors='coerce')
            except Exception:
                temp = pd.to_datetime(df[col], errors='coerce')
            if temp.notnull().sum() > 0:
                sorted_df = df.sort_values(by=col).reset_index(drop=True)
                sorted_time_col = col
                break
        except Exception:
            continue
    if sorted_time_col:
        st.info(f"已自动按时间列 {sorted_time_col} 升序排序显示")
    show_rows = st.slider("选择表格显示的行数", min_value=5, max_value=min(100, len(sorted_df)), value=10, step=1)
    st.dataframe(sorted_df.head(show_rows))
    st.write(f"总行数：{len(sorted_df)}，总列数：{len(sorted_df.columns)}")
    st.write("缺失值统计：")
    na_count = sorted_df.isnull().sum().to_frame(name="缺失值个数")
    na_percent = (sorted_df.isnull().mean() * 100).round(2).astype(str) + '%'
    na_percent = na_percent.to_frame(name="缺失值百分比%")
    na_stat = pd.concat([na_count, na_percent], axis=1)
    na_stat.index.name = None
    st.dataframe(na_stat.style.set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center')]},
        {'selector': 'td', 'props': [('text-align', 'center')]}
    ]), height=300)

    df = sorted_df
    # 不要在这里 st.session_state.data_cleaning_df = df

    # 数据类型选择
    st.subheader("🧼 异常值处理")
    data_type = st.radio("请选择数据类型", ["非时序数值", "时序数值"], horizontal=True)
    time_col = None
    if data_type == "时序数值":
        # 选择时间列并排序
        time_candidates = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])]
        if not time_candidates:
            st.warning("未检测到可用的时间列，请先在原始数据中添加时间列！")
            st.stop()
        time_col = st.selectbox("选择时间列（将自动升序排序）", time_candidates, key="time_col")
        # 尝试将时间列转为datetime
        try:
            df[time_col] = pd.to_datetime(df[time_col])
        except Exception:
            st.warning(f"{time_col} 无法自动转换为时间格式，请检查数据！")
            st.stop()
        df = df.sort_values(by=time_col).reset_index(drop=True)
        st.session_state.data_cleaning_df = df

    # 选择需要异常值处理的列
    selected_cols_outlier = st.multiselect("选择需要进行异常值处理的列", [col for col in df.columns if col != time_col], key="outlier_cols")
    outlier_methods = {}
    # 仅时序数据显示滑动窗口参数
    if data_type == "时序数值":
        window_size = st.number_input("滑动窗口大小（用于部分方法）", min_value=3, max_value=100, value=5, step=1)
    if "outlier_last_affected" not in st.session_state:
        st.session_state.outlier_last_affected = 0
    for col in selected_cols_outlier:
        st.markdown(f"**列：{col}**")
        if data_type == "时序数值":
            outlier_choices = [
                "不处理", "滑动窗口3σ法", "滑动窗口IQR法", "Z-score法", "自定义阈值"
            ]
        else:
            outlier_choices = [
                "不处理", "全局3σ法", "全局IQR法", "Z-score法", "自定义阈值"
            ]
        outlier_method = st.selectbox(
            f"选择异常值识别方式 - {col}",
            outlier_choices,
            key=f"outlier_{col}"
        )
        lower, upper = None, None
        if outlier_method == "Z-score法":
            zscore_thresh = st.number_input(f"Z-score阈值 - {col}", min_value=1.0, max_value=10.0, value=3.0, step=0.1, key=f"zscore_{col}")
        else:
            zscore_thresh = None
        if outlier_method == "自定义阈值":
            col_numeric = pd.to_numeric(df[col], errors='coerce')
            lower_default = float(col_numeric.min()) if not np.isnan(col_numeric.min()) else 0.0
            upper_default = float(col_numeric.max()) if not np.isnan(col_numeric.max()) else 0.0
            lower = st.number_input(f"下限 - {col}", value=lower_default, key=f"lower_{col}")
            upper = st.number_input(f"上限 - {col}", value=upper_default, key=f"upper_{col}")
        outlier_methods[col] = (outlier_method, lower, upper, zscore_thresh)
    if st.button("开始异常值处理", key="btn_outlier") and selected_cols_outlier:
        df_new = df.copy()
        # 自动将选中列转为数值型，无法转换的变为NaN
        for col in selected_cols_outlier:
            if pd.api.types.is_numeric_dtype(df_new[col]) or data_type == "非时序数值" or data_type == "时序数值":
                df_new[col] = pd.to_numeric(df_new[col], errors='coerce')
        changed_rows = set()
        for col in selected_cols_outlier:
            method, lower, upper, zscore_thresh = outlier_methods[col]
            series = df_new[col]
            before_na = series.isna().sum()
            # 时序异常值处理
            if data_type == "时序数值":
                if 'window_size' not in locals():
                    window_size = 5
                if method == "滑动窗口3σ法":
                    for i in range(len(series)):
                        left = max(0, i - window_size//2)
                        right = min(len(series), i + window_size//2 + 1)
                        window = series[left:right]
                        mean = window.mean()
                        std = window.std()
                        if abs(series.iloc[i] - mean) > 3*std:
                            df_new.at[series.index[i], col] = np.nan
                elif method == "滑动窗口IQR法":
                    for i in range(len(series)):
                        left = max(0, i - window_size//2)
                        right = min(len(series), i + window_size//2 + 1)
                        window = series[left:right]
                        q1 = window.quantile(0.25)
                        q3 = window.quantile(0.75)
                        iqr = q3 - q1
                        lower_iqr = q1 - 1.5 * iqr
                        upper_iqr = q3 + 1.5 * iqr
                        if series.iloc[i] < lower_iqr or series.iloc[i] > upper_iqr:
                            df_new.at[series.index[i], col] = np.nan
            # 非时序异常值处理
            if data_type == "非时序数值":
                if method == "全局3σ法":
                    mean = series.mean()
                    std = series.std()
                    mask = (series < mean - 3*std) | (series > mean + 3*std)
                    df_new.loc[mask, col] = np.nan
                elif method == "全局IQR法":
                    q1 = series.quantile(0.25)
                    q3 = series.quantile(0.75)
                    iqr = q3 - q1
                    lower_iqr = q1 - 1.5 * iqr
                    upper_iqr = q3 + 1.5 * iqr
                    mask = (series < lower_iqr) | (series > upper_iqr)
                    df_new.loc[mask, col] = np.nan
            # 通用异常值处理
            if method == "Z-score法" and zscore_thresh is not None:
                mean = series.me()
                std = series.std()
                z = (series - mean) / std
                df_new.loc[z.abs() > zscore_thresh, col] = np.nan
            elif method == "自定义阈值" and lower is not None and upper is not None:
                series_numeric = pd.to_numeric(series, errors='coerce')
                mask = (series_numeric < lower) | (series_numeric > upper)
                df_new.loc[mask, col] = np.nan
            after_na = df_new[col].isna().sum()
            changed = after_na - before_na
            if changed > 0:
                changed_rows.update(df_new.index[df_new[col].isna() & ~series.isna()])
        st.session_state.data_cleaning_df = df_new
        st.session_state.outlier_last_affected = len(changed_rows)
        st.session_state.outlier_show_flag = True
        st.experimental_rerun() if hasattr(st, 'experimental_rerun') else st.rerun()
    if st.session_state.get("outlier_show_flag", False):
        st.info(f"已处理")
        st.session_state.outlier_show_flag = False

    # 缺失值填充统计持久化
    if "fill_last_affected" not in st.session_state:
        st.session_state.fill_last_affected = 0
    if "fill_last_deleted" not in st.session_state:
        st.session_state.fill_last_deleted = 0
    # 缺失值填充区块
    st.subheader("🧽 缺失值填充")
    selected_cols_fill = st.multiselect("选择需要填充缺失值的列", [col for col in df.columns if col != time_col], key="fill_cols")
    fill_methods = {}
    # 仅时序数据显示多项式插值阶数
    if data_type == "时序数值":
        poly_order = st.number_input("多项式插值阶数", min_value=1, max_value=5, value=2, step=1)
    for col in selected_cols_fill:
        st.markdown(f"**列：{col}**")
        if data_type == "时序数值":
            fill_choices = [
                "不处理", "前向填充", "后向填充", "线性插值", "多项式插值", "窗口均值填充", "窗口中位数填充", "自定义值填充"
            ]
        else:
            fill_choices = [
                "不处理", "均值填充", "中位数填充", "众数填充", "自定义值填充"
            ]
        fill_method = st.selectbox(
            f"选择缺失值填充方式 - {col}",
            fill_choices,
            key=f"fill_{col}_fill"
        )
        fill_value = None
        if fill_method == "自定义值填充":
            fill_value = st.text_input(f"输入自定义填充值 - {col}", key=f"fillval_{col}_fill")
        fill_methods[col] = (fill_method, fill_value)
    # 删除指定列值为空的所有行
    del_na_cols = st.multiselect("选择要删除缺失值行的列（这些列有空值的行会被删除）", [col for col in df.columns if col != time_col], key="del_na_cols")
    if st.button("开始缺失值填充/删除", key="btn_fill") and (selected_cols_fill or del_na_cols):
        df_new = df.copy()
        # 自动将选中列转为数值型，无法转换的变为NaN
        for col in selected_cols_fill:
            if pd.api.types.is_numeric_dtype(df_new[col]) or data_type == "非时序数值" or data_type == "时序数值":
                df_new[col] = pd.to_numeric(df_new[col], errors='coerce')
        changed_rows_fill = set()
        # 先填充缺失值
        for col in selected_cols_fill:
            method, val = fill_methods[col]
            before_na = df_new[col].isna().sum()
            if data_type == "时序数值":
                if 'poly_order' not in locals():
                    poly_order = 2
                if method == "前向填充":
                    df_new[col] = df_new[col].fillna(method="ffill")
                elif method == "后向填充":
                    df_new[col] = df_new[col].fillna(method="bfill")
                elif method == "线性插值":
                    df_new[col] = df_new[col].interpolate(method="linear")
                elif method == "多项式插值":
                    df_new[col] = df_new[col].interpolate(method="polynomial", order=poly_order)
                elif method == "窗口均值填充":
                    window_size = 5 if 'window_size' not in locals() else window_size
                    df_new[col] = df_new[col].fillna(df_new[col].rolling(window=window_size, min_periods=1, center=True).mean())
                elif method == "窗口中位数填充":
                    window_size = 5 if 'window_size' not in locals() else window_size
                    df_new[col] = df_new[col].fillna(df_new[col].rolling(window=window_size, min_periods=1, center=True).median())
                elif method == "自定义值填充" and val is not None:
                    df_new[col] = df_new[col].fillna(val)
            else:
                if method == "均值填充":
                    df_new[col] = df_new[col].fillna(df_new[col].mean())
                elif method == "中位数填充":
                    df_new[col] = df_new[col].fillna(df_new[col].median())
                elif method == "众数填充":
                    df_new[col] = df_new[col].fillna(df_new[col].mode().iloc[0] if not df_new[col].mode().empty else np.nan)
                elif method == "自定义值填充" and val is not None:
                    df_new[col] = df_new[col].fillna(val)
            after_na = df_new[col].isna().sum()
            changed = before_na - after_na
            if changed > 0:
                changed_rows_fill.update(df_new.index[df_new[col].notna() & df[col].isna()])
        # 再删除指定列缺失值的行
        deleted_rows = 0
        if del_na_cols:
            before = len(df_new)
            df_new = df_new.dropna(subset=del_na_cols)
            after = len(df_new)
            deleted_rows = before - after
        st.session_state.data_cleaning_df = df_new
        st.session_state.fill_last_affected = len(changed_rows_fill)
        st.session_state.fill_last_deleted = deleted_rows
        st.experimental_rerun() if hasattr(st, 'experimental_rerun') else st.rerun()
    st.info(f"本次缺失值填充影响了 {st.session_state.get('fill_last_affected', 0)} 行，删除了 {st.session_state.get('fill_last_deleted', 0)} 行。")
    # ====== 导出按钮（缺失值填充后） ======
    export_format_fill = st.selectbox("选择导出格式", ["CSV", "Excel"], key="export_format_fill")
    if export_format_fill == "CSV":
        csv = st.session_state.data_cleaning_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("下载处理后数据 (CSV)", csv, file_name="cleaned_data.csv", mime="text/csv")
    else:
        import io
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            st.session_state.data_cleaning_df.to_excel(writer, index=False)
        st.download_button("下载处理后数据 (Excel)", output.getvalue(), file_name="cleaned_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
