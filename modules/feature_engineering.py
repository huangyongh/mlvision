import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, PolynomialFeatures



def load_data():
    df = None
    file = st.file_uploader("上传数据文件（支持 CSV, Excel）", type=["csv", "xls", "xlsx"], key="feature_engineering_upload")
    # 优先从 session_state 读取
    if "feature_df" in st.session_state:
        df = st.session_state.feature_df
    # 如果有新上传的文件，尝试读取
    if file is not None:
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.name.endswith('.xls') or file.name.endswith('.xlsx'):
                df = pd.read_excel(file)
            else:
                st.error("仅支持CSV或Excel文件！")
                return None
            st.session_state.feature_df = df.copy()
            st.success("文件上传并读取成功！")
        except Exception as e:
            st.error(f"文件读取失败: {e}")
            return None
    if df is None:
        # st.info("请上传数据或先在数据清洗模块导入数据")
        return None
    return df

def run():
    st.header("🧬 特征工程模块")
    # 始终以 session_state.feature_df 作为唯一数据源
    if "feature_df" in st.session_state:
        df = st.session_state.feature_df
    else:
        df = load_data()
        if df is not None:
            st.session_state.feature_df = df
    if "feature_df" not in st.session_state or st.session_state.feature_df is None:
        st.info("请上传数据或先在数据清洗模块导入数据")
        return
    df = st.session_state.feature_df
    st.write(f"数据维度：{df.shape[0]} 行，{df.shape[1]} 列")
    show_rows = st.slider("显示前N行", 5, min(100, len(df)), 10)
    st.dataframe(df.head(show_rows))

    # 2. 特征选择
    st.subheader("🔎 特征选择")
    with st.expander("相关性分析"):
        corr_method = st.selectbox("相关性方法", ["pearson", "spearman", "kendall"])
        # 更严格识别时间字段：datetime64 或 object 且全部能转为时间
        time_cols = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                time_cols.append(col)
            elif pd.api.types.is_object_dtype(df[col]):
                try:
                    converted = pd.to_datetime(df[col], errors='raise')
                    time_cols.append(col)
                except Exception:
                    pass
        # 只保留真正的数值型且非全NaN字段
        num_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col not in time_cols]
        valid_num_cols = [col for col in num_cols if not df[col].isnull().all()]
        if time_cols:
            st.info(f"以下时间字段已自动排除相关性分析: {time_cols}")
        if valid_num_cols:
            corr = df[valid_num_cols].corr(method=corr_method)
            st.dataframe(corr.style.background_gradient(cmap='coolwarm'))
        else:
            st.warning("无可用于相关性分析的数值型字段！")
    with st.expander("方差过滤"):
        var_thresh = st.number_input("最小方差阈值", 0.0, 10.0, 0.01, 0.01)
        low_var_cols = [col for col in df.select_dtypes(include=[np.number]).columns if df[col].var() < var_thresh]
        st.write(f"低方差特征: {low_var_cols}")
        if st.button("删除低方差特征") and low_var_cols:
            df = df.drop(columns=low_var_cols)
            st.session_state.feature_df = df
            st.success(f"已删除: {low_var_cols}")
            st.experimental_rerun() if hasattr(st, 'experimental_rerun') else st.rerun()
    with st.expander("缺失率过滤"):
        na_thresh = st.slider("最大缺失率(%)", 0, 100, 50)
        high_na_cols = [col for col in df.columns if df[col].isnull().mean()*100 > na_thresh]
        st.write(f"高缺失率特征: {high_na_cols}")
        if st.button("删除高缺失率特征") and high_na_cols:
            df = df.drop(columns=high_na_cols)
            st.session_state.feature_df = df
            st.success(f"已删除: {high_na_cols}")
            st.experimental_rerun() if hasattr(st, 'experimental_rerun') else st.rerun()
    with st.expander("手动选择/删除特征"):
        drop_cols = st.multiselect("选择要删除的特征", df.columns.tolist(), key="manual_drop")
        if st.button("手动删除特征") and drop_cols:
            df = df.drop(columns=drop_cols)
            st.session_state.feature_df = df
            st.success(f"已删除: {drop_cols}")
            st.experimental_rerun() if hasattr(st, 'experimental_rerun') else st.rerun()

    # 3. 特征变换
    st.subheader("🔄 特征变换")
    with st.expander("标准化/归一化"):
        num_cols = st.multiselect("选择数值列", df.select_dtypes(include=[np.number]).columns.tolist(), key="scale_cols")
        scale_method = st.selectbox("方法", ["不处理", "标准化(StandardScaler)", "归一化(MinMaxScaler)"])
        if st.button("应用变换", key="scale_btn") and num_cols and scale_method != "不处理":
            if scale_method == "标准化(StandardScaler)":
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler()
            df[num_cols] = scaler.fit_transform(df[num_cols])
            st.session_state.feature_df = df
            st.success(f"已对 {num_cols} 应用 {scale_method}")
            st.experimental_rerun() if hasattr(st, 'experimental_rerun') else st.rerun()
    with st.expander("编码"):
        cat_cols = st.multiselect("选择类别列", df.select_dtypes(include=['object', 'category']).columns.tolist(), key="cat_cols")
        encode_method = st.selectbox("编码方式", ["不处理", "OneHotEncoder", "LabelEncoder"])
        if st.button("应用编码", key="encode_btn") and cat_cols and encode_method != "不处理":
            if encode_method == "OneHotEncoder":
                df = pd.get_dummies(df, columns=cat_cols)
            else:
                for col in cat_cols:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
            st.session_state.feature_df = df
            st.success(f"已对 {cat_cols} 应用 {encode_method}")
            st.experimental_rerun() if hasattr(st, 'experimental_rerun') else st.rerun()
    with st.expander("数值分箱"):
        bin_col = st.selectbox("选择分箱列", df.select_dtypes(include=[np.number]).columns.tolist(), key="bin_col")
        n_bins = st.slider("分箱数", 2, 20, 5)
        if st.button("应用分箱", key="bin_btn") and bin_col:
            df[f"{bin_col}_bin"] = pd.cut(df[bin_col], bins=n_bins, labels=False)
            st.session_state.feature_df = df
            st.success(f"已对 {bin_col} 分箱，生成新特征 {bin_col}_bin")
            st.experimental_rerun() if hasattr(st, 'experimental_rerun') else st.rerun()
    with st.expander("多项式特征"):
        poly_cols = st.multiselect("选择生成多项式特征的列", df.select_dtypes(include=[np.number]).columns.tolist(), key="poly_cols")
        poly_degree = st.slider("多项式阶数", 2, 5, 2)
        if st.button("生成多项式特征", key="poly_btn") and poly_cols:
            pf = PolynomialFeatures(degree=poly_degree, include_bias=False)
            poly_data = pf.fit_transform(df[poly_cols])
            poly_names = pf.get_feature_names_out(poly_cols)
            poly_df = pd.DataFrame(poly_data, columns=poly_names, index=df.index)
            df = pd.concat([df, poly_df.iloc[:, len(poly_cols):]], axis=1)
            st.session_state.feature_df = df
            st.success(f"已生成多项式特征: {list(poly_names[len(poly_cols):])}")
            st.experimental_rerun() if hasattr(st, 'experimental_rerun') else st.rerun()

    # 4. 特征构造
    st.subheader("🛠️ 特征构造")
    with st.expander("特征交互"):
        inter_cols = st.multiselect("选择要交互的列（两两相乘）", df.select_dtypes(include=[np.number]).columns.tolist(), key="inter_cols")
        if st.button("生成交互特征", key="inter_btn") and len(inter_cols) >= 2:
            new_cols = []
            for i in range(len(inter_cols)):
                for j in range(i+1, len(inter_cols)):
                    new_col = f"{inter_cols[i]}_x_{inter_cols[j]}"
                    df[new_col] = df[inter_cols[i]] * df[inter_cols[j]]
                    new_cols.append(new_col)
            st.session_state.feature_df = df
            if 'new_feature_cols' not in st.session_state:
                st.session_state['new_feature_cols'] = []
            st.session_state['new_feature_cols'].extend([col for col in new_cols if col not in st.session_state['new_feature_cols']])
            if new_cols:
                st.success(f"已生成交互特征: {new_cols}")
                st.dataframe(df[new_cols].head(), use_container_width=True)
            else:
                st.info("未生成新的交互特征，请选择两列及以上。")
        # 不再 rerun，直接展示结果

    with st.expander("时间特征提取"):
        time_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])]
        time_col = st.selectbox("选择时间列", time_cols, key="time_feat_col") if time_cols else None
        if time_col:
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            if st.button("提取时间特征", key="time_feat_btn"):
                new_time_cols = [f"{time_col}_year", f"{time_col}_month", f"{time_col}_day", f"{time_col}_weekday"]
                df[f"{time_col}_year"] = df[time_col].dt.year
                df[f"{time_col}_month"] = df[time_col].dt.month
                df[f"{time_col}_day"] = df[time_col].dt.day
                df[f"{time_col}_weekday"] = df[time_col].dt.weekday
                st.session_state.feature_df = df
                if 'new_time_cols' not in st.session_state:
                    st.session_state['new_time_cols'] = []
                st.session_state['new_time_cols'].extend([col for col in new_time_cols if col not in st.session_state['new_time_cols']])
                st.success(f"已提取时间特征: {new_time_cols}")
                st.dataframe(df[new_time_cols].head(), use_container_width=True)

    with st.expander("滞后特征/滑动窗口"):
        ts_cols = st.multiselect("选择时序列", df.select_dtypes(include=[np.number]).columns.tolist(), key="lag_cols")
        lag_num = st.slider("滞后步数", 1, 10, 1)
        win_size = st.slider("滑动窗口大小", 2, 20, 3)
        if st.button("生成滞后/滑动窗口特征", key="lag_btn") and ts_cols:
            new_lag_cols = []
            for col in ts_cols:
                lag_col = f"{col}_lag{lag_num}"
                roll_col = f"{col}_rollmean{win_size}"
                df[lag_col] = df[col].shift(lag_num)
                df[roll_col] = df[col].rolling(window=win_size, min_periods=1).mean()
                new_lag_cols.extend([lag_col, roll_col])
            st.session_state.feature_df = df
            if 'new_lag_cols' not in st.session_state:
                st.session_state['new_lag_cols'] = []
            st.session_state['new_lag_cols'].extend([col for col in new_lag_cols if col not in st.session_state['new_lag_cols']])
            st.success(f"已生成滞后和滑动窗口特征: {new_lag_cols}")
            st.dataframe(df[new_lag_cols].head(), use_container_width=True)

    # 页面下方统一展示所有新特征
    st.markdown("---")
    show_new_cols = []
    if 'new_feature_cols' in st.session_state:
        show_new_cols.extend([col for col in st.session_state['new_feature_cols'] if col in df.columns])
    if 'new_time_cols' in st.session_state:
        show_new_cols.extend([col for col in st.session_state['new_time_cols'] if col in df.columns])
    if 'new_lag_cols' in st.session_state:
        show_new_cols.extend([col for col in st.session_state['new_lag_cols'] if col in df.columns])
    if show_new_cols:
        st.subheader("🆕 新生成特征一览")
        st.write(f"新特征列: {show_new_cols}")
        st.dataframe(df[show_new_cols].head(10), use_container_width=True)

    # 5. 特征导出
    st.subheader("💾 特征导出")
    export_format = st.selectbox("选择导出格式", ["CSV", "Excel"], key="export_format")
    export_df = st.session_state.feature_df if 'feature_df' in st.session_state else df
    if export_format == "CSV":
        csv = export_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("下载特征数据 (CSV)", data=csv, file_name="features.csv", mime="text/csv")
    elif export_format == "Excel":
        import io
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            export_df.to_excel(writer, index=False, sheet_name='Sheet1')
        st.download_button("下载特征数据 (Excel)", data=output.getvalue(), file_name="features.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

