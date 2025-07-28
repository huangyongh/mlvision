import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def run():
    st.header("🔬 机器学习分类建模（数值型数据）")
    # 只允许上传文件作为数据源
    file = st.file_uploader("上传数据文件（支持 CSV, Excel）", type=["csv", "xls", "xlsx"], key="clf_upload")
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
        label_col = st.selectbox("选择目标（标签）列", all_cols, key="clf_label")
        feature_cols = st.multiselect("选择特征列（可多选）", [c for c in all_cols if c != label_col], default=[c for c in all_cols if c != label_col], key="clf_feats")
        if not feature_cols or not label_col:
            st.warning("请至少选择一个特征和一个标签！")
            st.stop()

    # 标签分布检查
    y = df[label_col]
    st.write('标签分布：')
    st.write(y.value_counts())
    if y.value_counts().min() < 2:
        st.error("标签中有类别样本数小于2，无法进行分层采样和交叉验证，请检查数据！")
        st.stop()

    # 3. 训练/测试集划分
    with st.expander("2️⃣ 训练/测试集划分"):
        test_size = st.slider("测试集比例", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("随机种子", 0, 9999, 42)
        X = df[feature_cols]
        # y = df[label_col]  # 已提前定义
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        st.write(f"训练集：{X_train.shape}，测试集：{X_test.shape}")

    # 4. 模型选择
    with st.expander("3️⃣ 模型选择"):
        model_dict = {
            "逻辑回归": LogisticRegression,
            "K近邻": KNeighborsClassifier,
            "决策树": DecisionTreeClassifier,
            "随机森林": RandomForestClassifier,
            "梯度提升树": GradientBoostingClassifier,
            "XGBoost": XGBClassifier,
            "LightGBM": LGBMClassifier,
            "支持向量机": SVC,
            "朴素贝叶斯": GaussianNB,
        }
        model_names = list(model_dict.keys())
        selected_models = st.multiselect("选择要对比的模型（可多选）", model_names, default=["逻辑回归", "随机森林"])
        if not selected_models:
            st.warning("请至少选择一个模型！")
            st.stop()

    # 4.1 每个模型参数设置
    model_params = {}
    for m in selected_models:
        with st.expander(f"{m} 参数设置"):
            params = {}
            if m == "逻辑回归":
                C = st.number_input(f"C（正则化强度，越小正则越强）", 0.001, 100.0, 1.0, key=f"{m}_C")
                solver = st.selectbox(f"solver（优化器）", ["lbfgs", "liblinear", "saga"], key=f"{m}_solver")
                max_iter = st.number_input(f"max_iter（最大迭代次数）", 50, 5000, 1000, key=f"{m}_max_iter")
                params = {"C": C, "solver": solver, "max_iter": max_iter}
            elif m == "K近邻":
                n_neighbors = st.number_input(f"n_neighbors（邻居数）", 1, 50, 5, key=f"{m}_n_neighbors")
                weights = st.selectbox(f"weights", ["uniform", "distance"], key=f"{m}_weights")
                params = {"n_neighbors": n_neighbors, "weights": weights}
            elif m == "决策树":
                max_depth = st.number_input(f"max_depth（最大深度）", 1, 50, 5, key=f"{m}_max_depth")
                criterion = st.selectbox(f"criterion", ["gini", "entropy", "log_loss"], key=f"{m}_criterion")
                params = {"max_depth": max_depth, "criterion": criterion}
            elif m == "随机森林":
                n_estimators = st.number_input(f"n_estimators（树数）", 10, 1000, 100, key=f"{m}_n_estimators")
                max_depth = st.number_input(f"max_depth（最大深度）", 1, 50, 5, key=f"{m}_max_depth")
                params = {"n_estimators": n_estimators, "max_depth": max_depth}
            elif m == "梯度提升树":
                n_estimators = st.number_input(f"n_estimators（树数）", 10, 1000, 100, key=f"{m}_gbdt_n_estimators")
                learning_rate = st.number_input(f"learning_rate（学习率）", 0.001, 1.0, 0.1, key=f"{m}_learning_rate")
                params = {"n_estimators": n_estimators, "learning_rate": learning_rate}
            elif m == "XGBoost":
                n_estimators = st.number_input(f"n_estimators（树数）", 10, 1000, 100, key=f"{m}_xgb_n_estimators")
                learning_rate = st.number_input(f"learning_rate（学习率）", 0.001, 1.0, 0.1, key=f"{m}_xgb_learning_rate")
                use_label_encoder = False
                eval_metric = 'logloss'
                params = {"n_estimators": n_estimators, "learning_rate": learning_rate, "use_label_encoder": use_label_encoder, "eval_metric": eval_metric}
            elif m == "LightGBM":
                n_estimators = st.number_input(f"n_estimators（树数）", 10, 1000, 100, key=f"{m}_lgb_n_estimators")
                learning_rate = st.number_input(f"learning_rate（学习率）", 0.001, 1.0, 0.1, key=f"{m}_lgb_learning_rate")
                params = {"n_estimators": n_estimators, "learning_rate": learning_rate}
            elif m == "支持向量机":
                C = st.number_input(f"C（正则化强度）", 0.001, 100.0, 1.0, key=f"{m}_svm_C")
                kernel = st.selectbox(f"kernel（核函数）", ["rbf", "linear", "poly", "sigmoid"], key=f"{m}_svm_kernel")
                probability = True
                params = {"C": C, "kernel": kernel, "probability": probability}
            elif m == "朴素贝叶斯":
                var_smoothing = st.number_input(f"var_smoothing（方差平滑）", 1e-12, 1e-6, 1e-9, format="%e", key=f"{m}_nb_var_smoothing")
                params = {"var_smoothing": var_smoothing}
            model_params[m] = params

    # 5. 交叉验证与自动调参
    with st.expander("4️⃣ 交叉验证与自动调参"):
        use_cv = st.checkbox("启用交叉验证", value=True)
        cv_folds = st.slider("交叉验证折数", 3, 10, 5) if use_cv else 1
        use_search = st.checkbox("启用自动调参（GridSearchCV）", value=False)
        param_grid = {}
        if use_search:
            st.info("如需自动调参，请为每个模型设置参数范围（如 n_estimators: 50,100,200）")
            for m in selected_models:
                if m == "随机森林":
                    param_grid[m] = {"n_estimators": st.text_input("随机森林 n_estimators", "100,200,300").split(","),
                                     "max_depth": st.text_input("随机森林 max_depth", "3,5,7").split(",")}
                elif m == "逻辑回归":
                    param_grid[m] = {"C": st.text_input("逻辑回归 C", "0.1,1,10").split(",")}
                elif m == "K近邻":
                    param_grid[m] = {"n_neighbors": st.text_input("K近邻 n_neighbors", "3,5,7").split(",")}
                elif m == "XGBoost":
                    param_grid[m] = {"n_estimators": st.text_input("XGBoost n_estimators", "100,200").split(",")}
                elif m == "LightGBM":
                    param_grid[m] = {"n_estimators": st.text_input("LightGBM n_estimators", "100,200").split(",")}
                elif m == "支持向量机":
                    param_grid[m] = {"C": st.text_input("SVM C", "0.1,1,10").split(",")}
                elif m == "决策树":
                    param_grid[m] = {"max_depth": st.text_input("决策树 max_depth", "3,5,7").split(",")}
                elif m == "梯度提升树":
                    param_grid[m] = {"n_estimators": st.text_input("GBDT n_estimators", "100,200").split(",")}
                elif m == "朴素贝叶斯":
                    param_grid[m] = {}  # 通常无参数

    # 6. 训练与评估
    if st.button("开始训练并对比"):
        results = {}
        fig_roc, ax_roc = plt.subplots()
        progress_container = st.empty()
        info_container = st.empty()
        total = len(selected_models)
        # 新增：收集各模型主要指标
        metrics_dict = {}
        auc_dict = {}
        test_pred_figs = []
        for idx, m in enumerate(selected_models):
            info_container.info(f"正在训练：{m}")
            # 用用户设置的参数实例化模型
            model = model_dict[m](**model_params[m])
            best_params = None
            if use_search and m in param_grid and param_grid[m]:
                info_container.info(f"正在自动调参：{m} (GridSearchCV)")
                grid = {k: [float(x) if x.replace('.', '', 1).isdigit() else x for x in v if x] for k, v in param_grid[m].items()}
                search = GridSearchCV(model, grid, cv=cv_folds, scoring='roc_auc', n_jobs=-1)
                search.fit(X_train, y_train)
                model = search.best_estimator_
                best_params = search.best_params_
                st.write(f"最优参数: {best_params}")
            else:
                model.fit(X_train, y_train)
            # 交叉验证
            if use_cv:
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
                st.write(f"交叉验证准确率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            # 测试集评估
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            acc = accuracy_score(y_test, y_pred)
            # 分类报告表格化
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            # 主要指标收集
            metrics_dict[m] = {
                '准确率': acc,
                'F1分数': report_dict['weighted avg']['f1-score'],
                '召回率': report_dict['weighted avg']['recall'],
                '精确率': report_dict['weighted avg']['precision']
            }
            auc_value = None
            if y_proba is not None and len(np.unique(y_test)) == 2:
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                auc_value = auc(fpr, tpr)
                auc_dict[m] = auc_value
            else:
                auc_dict[m] = None
            # 测试集拟合曲线图片收集
            y_test_pred = y_pred
            fig_test, ax_test = plt.subplots()
            jitter = 0.1
            ax_test.scatter(y_test + np.random.uniform(-jitter, jitter, size=len(y_test)),
                            y_test_pred + np.random.uniform(-jitter, jitter, size=len(y_test_pred)),
                            alpha=0.5, color='orange', label='测试集', marker='x')
            ax_test.set_xlabel('真实值')
            ax_test.set_ylabel('预测值')
            ax_test.set_title(f'{m} 测试集拟合曲线（真实值-预测值）')
            ax_test.set_xticks(np.unique(y_test))
            ax_test.set_yticks(np.unique(y_test_pred))
            ax_test.legend()
            test_pred_figs.append(fig_test)
            # 其余展示（如2x3排版、模型保存等）保持不变
            # 2x3排版
            row1, row2 = st.columns(3), st.columns(3)
            # 混淆矩阵
            with row1[0]:
                fig_cm, ax_cm = plt.subplots()
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                ax_cm.set_xlabel('预测类别')
                ax_cm.set_ylabel('实际类别')
                ax_cm.set_title(f'{m} 混淆矩阵')
                st.pyplot(fig_cm)
            # 特征重要性
            with row1[1]:
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
                    fig_imp, ax_imp = plt.subplots()
                    feat_imp.plot(kind='bar', ax=ax_imp)
                    ax_imp.set_title(f'{m} 特征重要性')
                    ax_imp.set_ylabel('重要性')
                    ax_imp.set_xlabel('特征')
                    st.pyplot(fig_imp)
                elif hasattr(model, 'coef_'):
                    coef = model.coef_[0] if hasattr(model.coef_, '__len__') else model.coef_
                    feat_imp = pd.Series(coef, index=feature_cols).sort_values(ascending=False)
                    fig_imp, ax_imp = plt.subplots()
                    feat_imp.plot(kind='bar', ax=ax_imp)
                    ax_imp.set_title(f'{m} 系数')
                    ax_imp.set_ylabel('系数')
                    ax_imp.set_xlabel('特征')
                    st.pyplot(fig_imp)
                else:
                    st.info("该模型不支持特征重要性展示")
            # ROC曲线
            with row1[2]:
                if y_proba is not None and len(np.unique(y_test)) == 2:
                    fpr, tpr, _ = roc_curve(y_test, y_proba)
                    auc_value = auc(fpr, tpr)
                    fig_roc_single, ax_roc_single = plt.subplots()
                    ax_roc_single.plot(fpr, tpr, label=f'AUC={auc_value:.2f}')
                    ax_roc_single.plot([0, 1], [0, 1], 'k--')
                    ax_roc_single.set_xlabel('假正例率(FPR)')
                    ax_roc_single.set_ylabel('真正例率(TPR)')
                    ax_roc_single.set_title(f'{m} ROC曲线')
                    ax_roc_single.legend()
                    st.pyplot(fig_roc_single)
                else:
                    st.info("该模型不支持ROC曲线")
            # AUC
            with row2[0]:
                if auc_value is not None:
                    st.metric("AUC", f"{auc_value:.4f}")
                else:
                    st.info("无AUC")
            # 训练集拟合曲线
            with row2[1]:
                y_train_pred = model.predict(X_train)
                fig_train, ax_train = plt.subplots()
                idxs = np.arange(len(y_train))
                ax_train.scatter(idxs, y_train, alpha=0.5, label='真实值', marker='o', color='blue')
                ax_train.scatter(idxs, y_train_pred, alpha=0.5, label='预测值', marker='x', color='orange')
                ax_train.set_xlabel('样本编号')
                ax_train.set_ylabel('类别')
                ax_train.set_title(f'{m} 训练集拟合曲线（真实值-预测值）')
                ax_train.legend()
                st.pyplot(fig_train)
            # 测试集拟合曲线
            with row2[2]:
                y_test_pred = y_pred
                idxs = np.arange(len(y_test))
                fig_test, ax_test = plt.subplots()
                ax_test.scatter(idxs, y_test, alpha=0.5, label='真实值', marker='o', color='blue')
                ax_test.scatter(idxs, y_test_pred, alpha=0.5, label='预测值', marker='x', color='orange')
                ax_test.set_xlabel('样本编号')
                ax_test.set_ylabel('类别')
                ax_test.set_title(f'{m} 测试集拟合曲线（真实值-预测值）')
                ax_test.legend()
                st.pyplot(fig_test)

            # 保存模型
            results[m] = model
            progress_container.progress((idx + 1) / total, text=f"训练进度：{idx + 1}/{total}")
        progress_container.empty()
        info_container.empty()
        # 训练后统一展示主要指标表格
        st.subheader("各模型主要指标对比")
        metrics_df = pd.DataFrame(metrics_dict).T
        metrics_df['AUC'] = pd.Series(auc_dict)
        metrics_df = metrics_df[['准确率', 'F1分数', '召回率', '精确率', 'AUC']]
        metrics_df = metrics_df.replace({None: np.nan})
        st.dataframe(metrics_df.style.format({col: "{:.4f}" for col in metrics_df.select_dtypes(include=[np.number]).columns}), use_container_width=True)
        # 测试集拟合曲线图片区域，3列排布，与上面每个模型的测试集拟合曲线一致
        st.subheader("各模型测试集拟合曲线对比")
        for i, m in enumerate(selected_models):
            y_test_pred = results[m].predict(X_test)
            idxs = np.arange(len(y_test))
            fig, ax = plt.subplots()
            ax.scatter(idxs, y_test, alpha=0.5, label='真实值', marker='o', color='blue')
            ax.scatter(idxs, y_test_pred, alpha=0.5, label='预测值', marker='x', color='orange')
            ax.set_xlabel('样本编号')
            ax.set_ylabel('类别')
            ax.set_title(f'{m} 测试集拟合曲线（真实值-预测值）')
            ax.legend()
            if i % 3 == 0:
                cols = st.columns(3)
            with cols[i % 3]:
                st.pyplot(fig)

        # ROC曲线总览
        if len(selected_models) > 1 and ax_roc.has_data():
            ax_roc.plot([0, 1], [0, 1], 'k--')
            ax_roc.set_xlabel('假正例率(FPR)')
            ax_roc.set_ylabel('真正例率(TPR)')
            ax_roc.set_title('ROC曲线对比')
            ax_roc.legend()
            st.pyplot(fig_roc)
        # 模型导出
        st.subheader("模型导出")
        for m, model in results.items():
            buf = pickle.dumps(model)
            st.download_button(f"下载 {m} 模型", data=buf, file_name=f"{m}_model.pkl")
