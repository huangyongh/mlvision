import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import matplotlib
from sklearn.preprocessing import MinMaxScaler
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def run():
    st.header("⏳ 机器学习时序数据预测")
    # 1. 数据上传
    file = st.file_uploader("上传时序数据文件（支持 CSV, Excel）", type=["csv", "xls", "xlsx"], key="ts_upload")
    if file is not None:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith('.xls') or file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            st.error("仅支持CSV或Excel文件！")
            st.stop()
        st.success(f"已读取文件：{file.name}")
        st.dataframe(df.head())
    else:
        st.info("请上传数据文件")
        st.stop()

    # 2. 时间列选择
    with st.expander("1️⃣ 时间列选择"):
        time_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col]) or 'date' in col.lower() or 'time' in col.lower()]
        time_col = st.selectbox("选择时间列（必须唯一且递增）", time_cols if time_cols else df.columns, key="ts_time_col")
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.sort_values(by=time_col).reset_index(drop=True)
        st.write(f"数据按 {time_col} 升序排序")

    # 3. 特征/目标选择
    with st.expander("2️⃣ 特征与目标选择"):
        all_cols = [c for c in df.columns if c != time_col]
        label_col = st.selectbox("选择目标（标签）列", all_cols, key="ts_label")
        feature_cols = st.multiselect("选择特征列（可多选）", [c for c in all_cols if c != label_col], default=[c for c in all_cols if c != label_col], key="ts_feats")
        if not feature_cols or not label_col:
            st.warning("请至少选择一个特征和一个标签！")
            st.stop()

    # 3.1 归一化
    st.markdown("**特征与目标归一化（MinMaxScaler 0-1）**")
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    df[feature_cols] = scaler_X.fit_transform(df[feature_cols])
    df[label_col] = scaler_y.fit_transform(df[[label_col]])
    st.write("归一化后样例：")
    st.dataframe(df[feature_cols + [label_col]].head())

    # 3.1 预测类型与特征构建
    with st.expander("3️⃣ 预测类型与特征构建"):
        pred_type = st.selectbox("选择时序预测类型", [
            "单变量单步预测（目标变量滞后特征）",
            "多变量单步预测（目标变量滞后特征）",
            "多变量多步预测（目标变量滞后特征+多步标签）"
        ])
        lag_num = st.slider("滞后步数（lag）", 1, 24, 3)
        multi_step = 1
        if pred_type == "多变量多步预测（目标变量滞后特征+多步标签）":
            multi_step = st.slider("多步预测步数（未来步数）", 2, 12, 3)

    # 3.2 构建特征和标签
    def build_lagged_features_v2(df, feature_cols, label_col, lag_num, multi_step=1, single_var=False):
        data = {}
        # 当前时刻的所有特征
        for col in feature_cols:
            data[f'{col}_t'] = df[col]
        # 目标变量的前N步
        for l in range(1, lag_num+1):
            data[f'{label_col}_lag{l}'] = df[label_col].shift(l)
        # 标签
        if multi_step == 1:
            data['target'] = df[label_col].shift(-1)
        else:
            for s in range(1, multi_step+1):
                data[f'target_t+{s}'] = df[label_col].shift(-s)
        feat_df = pd.DataFrame(data)
        feat_df = feat_df.dropna().reset_index(drop=True)
        return feat_df

    # 构建特征和标签
    if pred_type == "单变量单步预测（目标变量滞后特征）":
        # 只用目标变量的前N步
        data = {}
        for l in range(1, lag_num+1):
            data[f'{label_col}_lag{l}'] = df[label_col].shift(l)
        data['target'] = df[label_col].shift(-1)
        feat_df = pd.DataFrame(data).dropna().reset_index(drop=True)
        target_col = 'target'
    elif pred_type == "多变量单步预测（目标变量滞后特征）":
        # 当前所有特征+目标变量的前N步，标签为目标变量t+1
        feat_df = build_lagged_features_v2(df, feature_cols, label_col, lag_num, 1, single_var=False)
        target_col = 'target'
    else:  # 多变量多步预测
        # 当前所有特征+目标变量的前N步，标签为目标变量t+1, t+2, ...
        feat_df = build_lagged_features_v2(df, feature_cols, label_col, lag_num, multi_step, single_var=False)
        target_cols = [f'target_t+{s}' for s in range(1, multi_step+1)]

    st.write("特征样例：")
    st.dataframe(feat_df.head())

    # 4. 按时间切分训练/测试集
    with st.expander("4️⃣ 训练/测试集划分（按时间顺序）"):
        test_size = st.slider("测试集比例", 0.05, 0.5, 0.2, 0.05)
        n = len(feat_df)
        n_test = int(n * test_size)
        n_train = n - n_test
        st.write(f"训练集：{n_train}，测试集：{n_test}")
        if pred_type == "多变量多步预测（目标变量滞后特征+多步标签）":
            X = feat_df.drop(columns=target_cols)
            y = feat_df[target_cols]
        else:
            X = feat_df.drop(columns=[target_col])
            y = feat_df[target_col]
        X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
        y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]
        # 时间索引
        time_test = df[time_col].iloc[-n_test: ] if n_test > 0 else df[time_col].iloc[-len(X_test):]

    # 5. 模型选择
    with st.expander("5️⃣ 模型选择"):
        model_dict = {
            "线性回归": LinearRegression,
            "岭回归": Ridge,
            "Lasso回归": Lasso,
            "弹性网回归": ElasticNet,
            "决策树回归": DecisionTreeRegressor,
            "随机森林回归": RandomForestRegressor,
            "梯度提升回归": GradientBoostingRegressor,
            "XGBoost回归": XGBRegressor,
            "LightGBM回归": LGBMRegressor,
            "支持向量回归": SVR,
        }
        model_names = list(model_dict.keys())
        selected_models = st.multiselect("选择要对比的模型（可多选）", model_names, default=["线性回归", "随机森林回归"])
        if not selected_models:
            st.warning("请至少选择一个模型！")
            st.stop()

    # 5.1 每个模型参数设置
    model_params = {}
    for m in selected_models:
        with st.expander(f"{m} 参数设置"):
            params = {}
            if m == "线性回归":
                fit_intercept = st.checkbox("fit_intercept", value=True, key=f"{m}_fit_intercept")
                params = {"fit_intercept": fit_intercept}
            elif m == "岭回归":
                alpha = st.number_input("alpha（正则化强度）", 0.0, 100.0, 1.0, key=f"{m}_alpha")
                fit_intercept = st.checkbox("fit_intercept", value=True, key=f"{m}_fit_intercept")
                max_iter = st.number_input("max_iter（最大迭代次数）", 10, 10000, 1000, key=f"{m}_max_iter")
                params = {"alpha": alpha, "fit_intercept": fit_intercept, "max_iter": max_iter}
            elif m == "Lasso回归":
                alpha = st.number_input("alpha（正则化强度）", 0.0, 100.0, 1.0, key=f"{m}_alpha")
                fit_intercept = st.checkbox("fit_intercept", value=True, key=f"{m}_fit_intercept")
                max_iter = st.number_input("max_iter（最大迭代次数）", 10, 10000, 1000, key=f"{m}_max_iter")
                params = {"alpha": alpha, "fit_intercept": fit_intercept, "max_iter": max_iter}
            elif m == "弹性网回归":
                alpha = st.number_input("alpha（正则化强度）", 0.0, 100.0, 1.0, key=f"{m}_alpha")
                l1_ratio = st.slider("l1_ratio", 0.0, 1.0, 0.5, key=f"{m}_l1_ratio")
                fit_intercept = st.checkbox("fit_intercept", value=True, key=f"{m}_fit_intercept")
                max_iter = st.number_input("max_iter（最大迭代次数）", 10, 10000, 1000, key=f"{m}_max_iter")
                params = {"alpha": alpha, "l1_ratio": l1_ratio, "fit_intercept": fit_intercept, "max_iter": max_iter}
            elif m == "决策树回归":
                max_depth = st.number_input("max_depth（最大深度）", 1, 50, 5, key=f"{m}_max_depth")
                params = {"max_depth": max_depth}
            elif m == "随机森林回归":
                n_estimators = st.number_input("n_estimators（树数）", 10, 1000, 100, key=f"{m}_n_estimators")
                max_depth = st.number_input("max_depth（最大深度）", 1, 50, 5, key=f"{m}_max_depth")
                params = {"n_estimators": n_estimators, "max_depth": max_depth}
            elif m == "梯度提升回归":
                n_estimators = st.number_input("n_estimators（树数）", 10, 1000, 100, key=f"{m}_gbdt_n_estimators")
                learning_rate = st.number_input("learning_rate（学习率）", 0.001, 1.0, 0.1, key=f"{m}_learning_rate")
                params = {"n_estimators": n_estimators, "learning_rate": learning_rate}
            elif m == "XGBoost回归":
                n_estimators = st.number_input("n_estimators（树数）", 10, 1000, 100, key=f"{m}_xgb_n_estimators")
                learning_rate = st.number_input("learning_rate（学习率）", 0.001, 1.0, 0.1, key=f"{m}_xgb_learning_rate")
                params = {"n_estimators": n_estimators, "learning_rate": learning_rate}
            elif m == "LightGBM回归":
                n_estimators = st.number_input("n_estimators（树数）", 10, 1000, 100, key=f"{m}_lgb_n_estimators")
                learning_rate = st.number_input("learning_rate（学习率）", 0.001, 1.0, 0.1, key=f"{m}_lgb_learning_rate")
                params = {"n_estimators": n_estimators, "learning_rate": learning_rate}
            elif m == "支持向量回归":
                C = st.number_input("C（正则化强度）", 0.001, 100.0, 1.0, key=f"{m}_svr_C")
                kernel = st.selectbox("kernel（核函数）", ["rbf", "linear", "poly", "sigmoid"], key=f"{m}_svr_kernel")
                max_iter = st.number_input("max_iter（最大迭代次数）", 10, 10000, 1000, key=f"{m}_svr_max_iter")
                params = {"C": C, "kernel": kernel, "max_iter": max_iter}
            model_params[m] = params

    # 6. 交叉验证（时序分割）
    with st.expander("6️⃣ 交叉验证"):
        use_cv = st.checkbox("启用交叉验证（TimeSeriesSplit）", value=True)
        cv_folds = st.slider("交叉验证折数", 3, 10, 5) if use_cv else 1

    # 7. 训练与评估
    if st.button("开始训练并对比"):
        results = {}
        metrics_dict = {}
        figs_pred = []
        figs_res = []
        figs_imp = []
        if pred_type == "多变量多步预测（目标变量滞后特征+多步标签）":
            # 多步预测：一次性多目标输出
            for m in selected_models:
                base_model = model_dict[m](**model_params[m])
                model = MultiOutputRegressor(base_model)
                model.fit(X_train, y_train)
                # 交叉验证（只对第一个目标）
                if use_cv:
                    tscv = TimeSeriesSplit(n_splits=cv_folds)
                    cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='r2')
                    st.write(f"[{m}] 交叉验证R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                y_pred = model.predict(X_test)  # shape: (n_samples, n_steps)
                # 评估和可视化：循环每个步长
                step_metrics = {}
                step_figs_pred = []
                step_figs_res = []
                step_figs_imp = []
                for i, step in enumerate(target_cols):
                    # 反归一化
                    y_pred_inv = scaler_y.inverse_transform(y_pred[:, i].reshape(-1, 1)).flatten()
                    y_test_inv = scaler_y.inverse_transform(y_test[step].values.reshape(-1, 1)).flatten()
                    mse = mean_squared_error(y_test_inv, y_pred_inv)
                    mae = mean_absolute_error(y_test_inv, y_pred_inv)
                    r2 = r2_score(y_test_inv, y_pred_inv)
                    step_metrics[step] = {"MSE": mse, "MAE": mae, "R²": r2}
                    # 1. 预测-真实值折线图（按时间）
                    fig_pred, ax_pred = plt.subplots()
                    ax_pred.plot(time_test, y_test_inv, label='真实值', marker='o', color='blue')
                    ax_pred.plot(time_test, y_pred_inv, label='预测值', marker='x', color='orange')
                    ax_pred.set_xlabel('时间')
                    ax_pred.set_ylabel('数值')
                    ax_pred.set_title(f'{m} {step} 测试集预测-真实值（按时间）')
                    ax_pred.legend()
                    step_figs_pred.append(fig_pred)
                    # 2. 残差分布图
                    fig_res, ax_res = plt.subplots()
                    residuals = y_test_inv - y_pred_inv
                    sns.histplot(residuals, bins=30, kde=True, ax=ax_res)
                    ax_res.set_title(f'{m} {step} 残差分布')
                    ax_res.set_xlabel('残差')
                    step_figs_res.append(fig_res)
                    # 3. 特征重要性或系数（取第一个子模型）
                    fig_imp = None
                    sub_est = model.estimators_[i] if hasattr(model, 'estimators_') else None
                    if sub_est is not None and hasattr(sub_est, 'feature_importances_'):
                        importances = sub_est.feature_importances_
                        feat_imp = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
                        fig_imp, ax_imp = plt.subplots()
                        feat_imp.plot(kind='bar', ax=ax_imp)
                        ax_imp.set_title(f'{m} {step} 特征重要性')
                        ax_imp.set_ylabel('重要性')
                        ax_imp.set_xlabel('特征')
                    elif sub_est is not None and hasattr(sub_est, 'coef_'):
                        coef = sub_est.coef_
                        if hasattr(coef, 'ndim') and coef.ndim == 1:
                            feat_imp = pd.Series(coef, index=X_train.columns).sort_values(ascending=False)
                        else:
                            feat_imp = pd.Series(coef[0], index=X_train.columns).sort_values(ascending=False)
                        fig_imp, ax_imp = plt.subplots()
                        feat_imp.plot(kind='bar', ax=ax_imp)
                        ax_imp.set_title(f'{m} {step} 系数')
                        ax_imp.set_ylabel('系数')
                        ax_imp.set_xlabel('特征')
                    step_figs_imp.append(fig_imp)
                # 每个模型每个步长单独展示
                st.subheader(f"每个步长详细信息以及可视化展示 - {m}")
                for i, step in enumerate(target_cols):
                    st.markdown(f"#### {step}")
                    r2 = step_metrics[step]["R²"]
                    mae = step_metrics[step]["MAE"]
                    mse = step_metrics[step]["MSE"]
                    st.write(f"R²: {r2:.4f}  |  MAE: {mae:.4f}  |  MSE: {mse:.4f}")
                    row = st.columns(3)
                    with row[0]:
                        st.pyplot(step_figs_pred[i])
                    with row[1]:
                        st.pyplot(step_figs_res[i])
                    with row[2]:
                        if step_figs_imp[i] is not None:
                            st.pyplot(step_figs_imp[i])
                        else:
                            st.info("该模型不支持特征重要性或系数展示")
                # 汇总表格
                st.subheader(f"各步长指标对比汇总 - {m}")
                metrics_df = pd.DataFrame(step_metrics).T
                metrics_df = metrics_df[["R²", "MAE", "MSE"]]
                st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)
                # 对比所有步长的测试集拟合曲线（预测-真实值折线图）
                st.subheader(f"各步长测试集预测-真实值对比（按时间） - {m}")
                n_steps = len(target_cols)
                for i in range(0, n_steps, 3):
                    cols = st.columns(3)
                    for j, idx in enumerate(range(i, min(i+3, n_steps))):
                        step = target_cols[idx]
                        with cols[j]:
                            st.markdown(f"**{step}**")
                            st.pyplot(step_figs_pred[idx])
                # 模型导出
                st.subheader(f"模型导出 - {m}")
                buf = pickle.dumps(model)
                st.download_button(f"下载 {m} 多步模型", data=buf, file_name=f"{m}_multioutput_ts_model.pkl")
        else:
            # 单步预测逻辑
            for m in selected_models:
                model = model_dict[m](**model_params[m])
                model.fit(X_train, y_train)
                # 交叉验证
                if use_cv:
                    tscv = TimeSeriesSplit(n_splits=cv_folds)
                    cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='r2')
                    st.write(f"[{m}] 交叉验证R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                # 测试集评估
                y_pred = model.predict(X_test)
                # 反归一化
                y_pred_inv = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                y_test_inv = scaler_y.inverse_transform(y_test.values.reshape(-1, 1)).flatten()
                mse = mean_squared_error(y_test_inv, y_pred_inv)
                mae = mean_absolute_error(y_test_inv, y_pred_inv)
                r2 = r2_score(y_test_inv, y_pred_inv)
                metrics_dict[m] = {"MSE": mse, "MAE": mae, "R²": r2}
                # 1. 预测-真实值折线图（按时间）
                fig_pred, ax_pred = plt.subplots()
                ax_pred.plot(time_test, y_test_inv, label='真实值', marker='o', color='blue')
                ax_pred.plot(time_test, y_pred_inv, label='预测值', marker='x', color='orange')
                ax_pred.set_xlabel('时间')
                ax_pred.set_ylabel('数值')
                ax_pred.set_title(f'{m} 测试集预测-真实值（按时间）')
                ax_pred.legend()
                figs_pred.append(fig_pred)
                # 2. 残差分布图
                fig_res, ax_res = plt.subplots()
                residuals = y_test_inv - y_pred_inv
                sns.histplot(residuals, bins=30, kde=True, ax=ax_res)
                ax_res.set_title(f'{m} 残差分布')
                ax_res.set_xlabel('残差')
                figs_res.append(fig_res)
                # 3. 特征重要性或系数
                fig_imp = None
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feat_imp = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
                    fig_imp, ax_imp = plt.subplots()
                    feat_imp.plot(kind='bar', ax=ax_imp)
                    ax_imp.set_title(f'{m} 特征重要性')
                    ax_imp.set_ylabel('重要性')
                    ax_imp.set_xlabel('特征')
                elif hasattr(model, 'coef_'):
                    coef = model.coef_
                    if hasattr(coef, 'ndim') and coef.ndim == 1:
                        feat_imp = pd.Series(coef, index=X_train.columns).sort_values(ascending=False)
                    else:
                        feat_imp = pd.Series(coef[0], index=X_train.columns).sort_values(ascending=False)
                    fig_imp, ax_imp = plt.subplots()
                    feat_imp.plot(kind='bar', ax=ax_imp)
                    ax_imp.set_title(f'{m} 系数')
                    ax_imp.set_ylabel('系数')
                    ax_imp.set_xlabel('特征')
                figs_imp.append(fig_imp)
                results[m] = model
            # 先每个模型单独展示所有指标和图
            st.subheader("每个模型详细信息以及可视化展示")
            for i, m in enumerate(selected_models):
                st.markdown(f"#### {m}")
                r2 = metrics_dict[m]["R²"]
                mae = metrics_dict[m]["MAE"]
                mse = metrics_dict[m]["MSE"]
                st.write(f"R²: {r2:.4f}  |  MAE: {mae:.4f}  |  MSE: {mse:.4f}")
                row = st.columns(3)
                with row[0]:
                    st.pyplot(figs_pred[i])
                with row[1]:
                    st.pyplot(figs_res[i])
                with row[2]:
                    if figs_imp[i] is not None:
                        st.pyplot(figs_imp[i])
                    else:
                        st.info("该模型不支持特征重要性或系数展示")
            # 最后统一汇总对比表格，行为模型，列为R²、MAE、MSE
            st.subheader("各模型指标对比汇总")
            metrics_df = pd.DataFrame(metrics_dict).T
            metrics_df = metrics_df[["R²", "MAE", "MSE"]]
            st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)
            # 最后对比所有模型的测试集拟合曲线（预测-真实值折线图）
            st.subheader("各模型测试集预测-真实值对比（按时间）")
            n_models = len(selected_models)
            for i in range(0, n_models, 3):
                cols = st.columns(3)
                for j, idx in enumerate(range(i, min(i+3, n_models))):
                    m = selected_models[idx]
                    with cols[j]:
                        st.markdown(f"**{m}**")
                        st.pyplot(figs_pred[idx])
        # 模型导出
        st.subheader("模型导出")
        for m, model in results.items():
            buf = pickle.dumps(model)
            st.download_button(f"下载 {m} 模型", data=buf, file_name=f"{m}_ts_model.pkl")
