import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def run():
    st.header("🔬 机器学习回归建模（数值型数据）")
    # 1. 数据上传
    file = st.file_uploader("上传数据文件（支持 CSV, Excel）", type=["csv", "xls", "xlsx"], key="reg_upload")
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

    # 2. 特征/标签选择
    with st.expander("1️⃣ 特征与标签选择"):
        all_cols = df.columns.tolist()
        label_col = st.selectbox("选择目标（标签）列", all_cols, key="reg_label")
        feature_cols = st.multiselect("选择特征列（可多选）", [c for c in all_cols if c != label_col], default=[c for c in all_cols if c != label_col], key="reg_feats")
        if not feature_cols or not label_col:
            st.warning("请至少选择一个特征和一个标签！")
            st.stop()

    # 3. 训练/测试集划分
    with st.expander("2️⃣ 训练/测试集划分"):
        test_size = st.slider("测试集比例", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("随机种子", 0, 9999, 42)
        X = df[feature_cols]
        y = df[label_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        st.write(f"训练集：{X_train.shape}，测试集：{X_test.shape}")

    # 4. 模型选择
    with st.expander("3️⃣ 模型选择"):
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

    # 4.1 每个模型参数设置
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

    # 5. 交叉验证
    with st.expander("4️⃣ 交叉验证"):
        use_cv = st.checkbox("启用交叉验证", value=True)
        cv_folds = st.slider("交叉验证折数", 3, 10, 5) if use_cv else 1

    # 6. 训练与评估
    if st.button("开始训练并对比"):
        results = {}
        metrics_dict = {}
        figs_pred = []
        figs_res = []
        figs_imp = []
        for m in selected_models:
            model = model_dict[m](**model_params[m])
            model.fit(X_train, y_train)
            # 交叉验证
            if use_cv:
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
                st.write(f"[{m}] 交叉验证R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            # 测试集评估
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            metrics_dict[m] = {"MSE": mse, "MAE": mae, "R²": r2}
            # 1. 预测-真实值散点图（x=真实值，y=预测值，真实值o蓝色，预测值x橙色，带图例）
            fig_pred, ax_pred = plt.subplots()
            idxs = np.arange(len(y_test))
            ax_pred.scatter(y_test, y_test, alpha=0.5, label='真实值', marker='o', color='blue')
            ax_pred.scatter(y_test, y_pred, alpha=0.5, label='预测值', marker='x', color='orange')
            ax_pred.set_xlabel('真实值')
            ax_pred.set_ylabel('预测值')
            ax_pred.set_title(f'{m} 测试集预测-真实值')
            ax_pred.legend()
            figs_pred.append(fig_pred)
            # 2. 残差分布图
            fig_res, ax_res = plt.subplots()
            residuals = y_test - y_pred
            sns.histplot(residuals, bins=30, kde=True, ax=ax_res)
            ax_res.set_title(f'{m} 残差分布')
            ax_res.set_xlabel('残差')
            figs_res.append(fig_res)
            # 3. 特征重要性或系数
            fig_imp = None
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
                fig_imp, ax_imp = plt.subplots()
                feat_imp.plot(kind='bar', ax=ax_imp)
                ax_imp.set_title(f'{m} 特征重要性')
                ax_imp.set_ylabel('重要性')
                ax_imp.set_xlabel('特征')
            elif hasattr(model, 'coef_'):
                coef = model.coef_
                if coef.ndim == 1:
                    feat_imp = pd.Series(coef, index=feature_cols).sort_values(ascending=False)
                else:
                    feat_imp = pd.Series(coef[0], index=feature_cols).sort_values(ascending=False)
                fig_imp, ax_imp = plt.subplots()
                feat_imp.plot(kind='bar', ax=ax_imp)
                ax_imp.set_title(f'{m} 系数')
                ax_imp.set_ylabel('系数')
                ax_imp.set_xlabel('特征')
            figs_imp.append(fig_imp)
            results[m] = model
        # 主要指标表格
        # st.subheader("各模型主要指标对比")
        # metrics_df = pd.DataFrame(metrics_dict)
        # st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)
        # 先每个模型单独展示所有指标和图
        st.subheader("每个模型详细信息以及可视化展示")
        for i, m in enumerate(selected_models):
            st.markdown(f"#### {m}")
            # 单独显示指标
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
        # 最后对比所有模型的测试集拟合曲线
        st.subheader("各模型测试集拟合曲线对比")
        n_models = len(selected_models)
        for i in range(0, n_models, 3):
            cols = st.columns(3)
            for j, idx in enumerate(range(i, min(i+3, n_models))):
                m = selected_models[idx]
                with cols[j]:
                    st.markdown(f"**{m}**")
                    st.pyplot(figs_pred[idx])
        
       
