import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from datetime import datetime
import io
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
import importlib.util
import tempfile
import yaml
import os

def run():
    # 1. 数据上传
    st.header("🧠 深度学习时序预测")
    file = st.file_uploader("上传时序数据文件（支持 CSV, Excel）", type=["csv", "xls", "xlsx"], key="dl_ts_upload")
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
        time_col = st.selectbox("选择时间列（必须唯一且递增）", time_cols if time_cols else df.columns, key="dl_ts_time_col")
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.sort_values(by=time_col).reset_index(drop=True)
        st.write(f"数据按 {time_col} 升序排序")

    # 3. 特征/目标选择
    with st.expander("2️⃣ 特征与目标选择"):
        all_cols = [c for c in df.columns if c != time_col]
        label_col = st.selectbox("选择目标（标签）列", all_cols, key="dl_ts_label")
        feature_cols = st.multiselect("选择特征列（可多选）", [c for c in all_cols if c != label_col], default=[c for c in all_cols if c != label_col], key="dl_ts_feats")
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

    # 4. 预测类型与窗口设置
    with st.expander("3️⃣ 预测类型与窗口设置"):
        pred_type = st.selectbox("选择时序预测类型", [
            "单变量单步预测（目标变量滞后特征）",
            "多变量单步预测（目标变量滞后特征）",
            "多变量多步预测（目标变量滞后特征+多步标签）"
        ])
        lag_num = st.slider("滞后步数（输入窗口长度）", 1, 48, 12)
        multi_step = 1
        if pred_type == "多变量多步预测（目标变量滞后特征+多步标签）":
            multi_step = st.slider("多步预测步数（输出步数）", 2, 24, 3)
        use_lag_target = st.checkbox("将目标变量的前N步作为额外特征", value=True)

    # 5. 构建深度学习输入数据
    st.subheader("数据预处理与窗口构建")
    def build_dl_windows(df, feature_cols, label_col, lag_num, multi_step=1, pred_type="单变量单步预测（目标变量滞后特征）", use_lag_target=True):
        X, y = [], []
        values = df[feature_cols + [label_col]].values
        label_values = df[label_col].values
        for i in range(lag_num, len(df) - multi_step + 1):
            if pred_type == "单变量单步预测（目标变量滞后特征）":
                X.append(label_values[i-lag_num:i].reshape(-1, 1))
                y.append(label_values[i + multi_step - 1])
            else:
                # 前N步所有特征 (N, 特征数)
                X_window = values[i-lag_num:i, :-1]
                if use_lag_target:
                    lagged_targets = label_values[i-lag_num:i].reshape(-1, 1)  # (N, 1)
                    X_full = np.concatenate([X_window, lagged_targets], axis=1)  # (N, 特征数+1)
                else:
                    X_full = X_window
                X.append(X_full)
                if multi_step == 1:
                    y.append(values[i, -1])
                else:
                    y.append([values[i + j, -1] for j in range(multi_step)])
        X = np.array(X)
        y = np.array(y)
        return X, y

    X, y = build_dl_windows(df, feature_cols, label_col, lag_num, multi_step, pred_type, use_lag_target)
    st.write(f"输入X shape: {X.shape}, 标签y shape: {y.shape}")

    # 预览窗口特征与标签
    st.markdown("**窗口特征与标签预览：**")
    col1, col2 = st.columns(2)
    with col1:
        st.write("特征窗口 (X[0]):")
        if pred_type == "单变量单步预测（目标变量滞后特征）":
            preview_cols = [label_col]
            st.dataframe(pd.DataFrame(X[0], columns=preview_cols))
        else:
            base_cols = feature_cols.copy()
            if use_lag_target:
                preview_cols = base_cols + [label_col]
            else:
                preview_cols = base_cols
            st.dataframe(pd.DataFrame(X[0], columns=preview_cols))
    with col2:
        st.write("标签 (y[0]):")
        if y.ndim == 1:
            st.write(y[0])
        else:
            st.write(y[0])

    # 6. 训练/测试集划分
    with st.expander("4️⃣ 训练/测试集划分（按时间顺序）"):
        test_size = st.slider("测试集比例", 0.05, 0.5, 0.2, 0.05)
        n = len(X)
        n_test = int(n * test_size)
        n_train = n - n_test
        st.write(f"训练集：{n_train}，测试集：{n_test}")
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]
        time_test = df[time_col].iloc[-n_test: ] if n_test > 0 else df[time_col].iloc[-len(X_test):]

    # 允许用户上传自定义模型.py文件
    st.markdown("**可选：上传自定义PyTorch模型（.py文件，需继承nn.Module，构造参数为(input_size, hidden_size, num_layers, output_size, dropout, **kwargs)）**")
    uploaded_model_file = st.file_uploader("上传自定义模型.py文件", type=["py"], key="custom_model_upload")
    uploaded_models = {}
    if uploaded_model_file is not None:
        custom_model_dir = os.path.join(os.path.dirname(__file__), "custom_models")
        os.makedirs(custom_model_dir, exist_ok=True)
        save_path = os.path.join(custom_model_dir, uploaded_model_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_model_file.read())
        module_name = os.path.splitext(uploaded_model_file.name)[0]
        spec = importlib.util.spec_from_file_location(module_name, save_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for attr in dir(module):
            cls = getattr(module, attr)
            if isinstance(cls, type) and issubclass(cls, nn.Module) and cls is not nn.Module:
                uploaded_models[attr] = cls
        st.success(f"已加载自定义模型: {', '.join(uploaded_models.keys())}")

    # 允许用户上传自定义模型参数yaml文件
    st.markdown("**可选：上传自定义模型参数（.yaml文件，内容为模型构造参数字典）**")
    uploaded_yaml_file = st.file_uploader("上传自定义模型参数.yaml文件", type=["yaml", "yml"], key="custom_model_yaml_upload")
    yaml_params = {}
    if uploaded_yaml_file is not None:
        yaml_params = yaml.safe_load(uploaded_yaml_file)
        st.info(f"已加载自定义参数: {yaml_params}")

    # 自动加载custom_models目录下模型
    custom_model_dir = os.path.join(os.path.dirname(__file__), "custom_models")
    custom_models = {}
    if os.path.exists(custom_model_dir):
        for fname in os.listdir(custom_model_dir):
            if fname.endswith(".py"):
                module_name = fname[:-3]
                spec = importlib.util.spec_from_file_location(module_name, os.path.join(custom_model_dir, fname))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                for attr in dir(module):
                    cls = getattr(module, attr)
                    if isinstance(cls, type) and issubclass(cls, nn.Module) and cls is not nn.Module:
                        custom_models[attr] = cls

    # 7. 模型选择与参数设置
    with st.expander("5️⃣ 模型选择与参数设置"):
        builtin_models = [
            "LSTM",
            "GRU",
            "MLP",
            "1D-CNN",
            "Transformer"
        ]
        model_choices = builtin_models + list(custom_models.keys()) + list(uploaded_models.keys())
        model_type = st.selectbox("选择深度学习模型结构", model_choices)
        hidden_size = st.slider("隐藏单元数/通道数", 8, 256, 64)
        num_layers = st.slider("层数", 1, 4, 1)
        dropout = st.slider("Dropout", 0.0, 0.5, 0.1)
        epochs = st.slider("训练轮数", 5, 200, 30)
        batch_size = st.slider("Batch size", 8, 256, 32)
        lr = st.number_input("学习率", 1e-5, 1e-1, 1e-3, format="%e")
        # 新增损失函数和优化器选择
        loss_options = {"MSELoss": nn.MSELoss, "L1Loss (MAE)": nn.L1Loss, "HuberLoss": nn.SmoothL1Loss, "CrossEntropyLoss (仅分类)": nn.CrossEntropyLoss}
        optimizer_options = {"Adam": optim.Adam, "SGD": optim.SGD, "RMSprop": optim.RMSprop, "Adagrad": optim.Adagrad}
        loss_choice = st.selectbox("选择损失函数", list(loss_options.keys()), index=0, help="CrossEntropyLoss 仅适用于分类任务，回归请勿选")
        optimizer_choice = st.selectbox("选择优化器", list(optimizer_options.keys()), index=0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.write(f"当前设备: {device}")

    # 8. 定义模型结构（可扩展）
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_size, output_size)
        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            out = self.fc(out)
            return out

    class GRUModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
            super().__init__()
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_size, output_size)
        def forward(self, x):
            out, _ = self.gru(x)
            out = out[:, -1, :]
            out = self.fc(out)
            return out

    class MLPModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
            super().__init__()
            layers = [nn.Flatten()]
            in_dim = input_size
            for _ in range(num_layers):
                layers.append(nn.Linear(in_dim, hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                in_dim = hidden_size
            layers.append(nn.Linear(hidden_size, output_size))
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x)

    class CNN1DModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
            super().__init__()
            layers = []
            in_channels = input_size
            for _ in range(num_layers):
                layers.append(nn.Conv1d(in_channels, hidden_size, kernel_size=3, padding=1))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                in_channels = hidden_size
            self.conv = nn.Sequential(*layers)
            self.fc = nn.Linear(hidden_size * lag_num, output_size)
        def forward(self, x):
            # x: (batch, seq, feat) -> (batch, feat, seq)
            x = x.permute(0, 2, 1)
            out = self.conv(x)
            out = out.reshape(out.size(0), -1)
            out = self.fc(out)
            return out

    class TransformerModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
            super().__init__()
            encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=4, dim_feedforward=hidden_size, dropout=dropout, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = nn.Linear(input_size, output_size)
        def forward(self, x):
            # x: (batch, seq, feat)
            out = self.transformer(x)
            out = out[:, -1, :]  # 取最后一个时间步
            out = self.fc(out)
            return out

    # 9. 训练与评估
    if st.button("开始训练"):
        st.info("数据准备中...")
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)
        train_ds = TensorDataset(X_train_t, y_train_t)
        test_ds = TensorDataset(X_test_t, y_test_t)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size)

        input_size = X_train.shape[2] if len(X_train.shape) == 3 else X_train.shape[1]
        output_size = y_train.shape[1] if len(y_train.shape) > 1 else 1
        if model_type == "LSTM":
            model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout).to(device)
        elif model_type == "GRU":
            model = GRUModel(input_size, hidden_size, num_layers, output_size, dropout).to(device)
        elif model_type == "MLP":
            model = MLPModel(input_size * lag_num, hidden_size, num_layers, output_size, dropout).to(device)
        elif model_type == "1D-CNN":
            model = CNN1DModel(input_size, hidden_size, num_layers, output_size, dropout).to(device)
        elif model_type == "Transformer":
            model = TransformerModel(input_size, hidden_size, num_layers, output_size, dropout).to(device)
        elif model_type in custom_models:
            model = custom_models[model_type](input_size, hidden_size, num_layers, output_size, dropout, **yaml_params).to(device)
        elif model_type in uploaded_models:
            model = uploaded_models[model_type](input_size, hidden_size, num_layers, output_size, dropout, **yaml_params).to(device)
        else:
            st.error("未知模型类型")
            st.stop()

        # 根据选择构建损失函数和优化器
        criterion = loss_options[loss_choice]()
        optimizer = optimizer_options[optimizer_choice](model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        st.info("开始训练...")
        progress_bar = st.progress(0)
        losses = []
        lrs = []
        train_scores = []
        test_scores = []
        # 动态可视化容器
        chart_row = st.columns(4)
        acc_chart = chart_row[0].empty()
        test_acc_chart = chart_row[1].empty()
        loss_chart = chart_row[2].empty()
        lr_chart = chart_row[3].empty()

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                if model_type == "MLP":
                    xb = xb.reshape(xb.size(0), -1)
                out = model(xb)
                loss = criterion(out, yb if output_size > 1 else yb.view(-1, 1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(train_loader.dataset)
            losses.append(epoch_loss)
            lrs.append(optimizer.param_groups[0]['lr'])
            # 训练准确度（R²）
            model.eval()
            with torch.no_grad():
                if model_type == "MLP":
                    X_train_eval = X_train_t.reshape(X_train_t.size(0), -1)
                    X_test_eval = X_test_t.reshape(X_test_t.size(0), -1)
                else:
                    X_train_eval = X_train_t
                    X_test_eval = X_test_t
                y_train_pred = model(X_train_eval).cpu().numpy()
                y_test_pred = model(X_test_eval).cpu().numpy()
                # 反归一化
                if y_train_pred.ndim == 1 or y_train_pred.shape[1] == 1:
                    y_train_pred_inv = scaler_y.inverse_transform(y_train_pred.reshape(-1, 1)).squeeze()
                    y_train_true_inv = scaler_y.inverse_transform(y_train.reshape(-1, 1)).squeeze()
                    y_test_pred_inv = scaler_y.inverse_transform(y_test_pred.reshape(-1, 1)).squeeze()
                    y_test_true_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).squeeze()
                else:
                    y_train_pred_inv = scaler_y.inverse_transform(y_train_pred)
                    y_train_true_inv = scaler_y.inverse_transform(y_train)
                    y_test_pred_inv = scaler_y.inverse_transform(y_test_pred)
                    y_test_true_inv = scaler_y.inverse_transform(y_test)
                if output_size == 1:
                    train_r2 = 1 - np.sum((y_train_true_inv - y_train_pred_inv) ** 2) / np.sum((y_train_true_inv - np.mean(y_train_true_inv)) ** 2)
                    test_r2 = 1 - np.sum((y_test_true_inv - y_test_pred_inv) ** 2) / np.sum((y_test_true_inv - np.mean(y_test_true_inv)) ** 2)
                else:
                    train_r2 = np.mean([1 - np.sum((y_train_true_inv[:, i] - y_train_pred_inv[:, i]) ** 2) / np.sum((y_train_true_inv[:, i] - np.mean(y_train_true_inv[:, i])) ** 2) for i in range(output_size)])
                    test_r2 = np.mean([1 - np.sum((y_test_true_inv[:, i] - y_test_pred_inv[:, i]) ** 2) / np.sum((y_test_true_inv[:, i] - np.mean(y_test_true_inv[:, i])) ** 2) for i in range(output_size)])
                train_scores.append(train_r2)
                test_scores.append(test_r2)
            scheduler.step(epoch_loss)
            # 动态刷新曲线
            with acc_chart:
                fig_train_acc, ax_train_acc = plt.subplots(figsize=(3, 2))
                ax_train_acc.plot(train_scores, label='训练R²')
                ax_train_acc.set_title('训练集R$^2$')
                ax_train_acc.set_xlim(0, epochs)
                ax_train_acc.set_ylim(0, 1)
                ax_train_acc.set_xlabel('Epoch')
                ax_train_acc.set_ylabel('R$^2$')
                acc_chart.pyplot(fig_train_acc)
            with test_acc_chart:
                fig_test_acc, ax_test_acc = plt.subplots(figsize=(3, 2))
                ax_test_acc.plot(test_scores, label='测试R²', color='orange')
                ax_test_acc.set_title('测试集R$^2$')
                ax_test_acc.set_xlim(0, epochs)
                ax_test_acc.set_ylim(0, 1)
                ax_test_acc.set_xlabel('Epoch')
                ax_test_acc.set_ylabel('R$^2$')
                test_acc_chart.pyplot(fig_test_acc)
            with loss_chart:
                fig_loss, ax_loss = plt.subplots(figsize=(3, 2))
                ax_loss.plot(losses)
                ax_loss.set_title('训练损失')
                ax_loss.set_xlabel('Epoch')
                ax_loss.set_ylabel('Loss')
                ax_loss.set_xlim(0, epochs)
                loss_chart.pyplot(fig_loss)
            with lr_chart:
                fig_lr, ax_lr = plt.subplots(figsize=(3, 2))
                ax_lr.plot(lrs)
                ax_lr.set_title('学习率')
                ax_lr.set_xlabel('Epoch')
                ax_lr.set_ylabel('Learning Rate')
                ax_lr.set_xlim(0, epochs)
                lr_chart.pyplot(fig_lr)
            progress_bar.progress((epoch+1)/epochs, text=f"训练进度: {epoch+1}/{epochs}")

        progress_bar.empty()
        st.success("训练完成！")

        # 训练/测试准确度曲线
        st.subheader("训练/测试集R²曲线")
        row_acc = st.columns(2)
        with row_acc[0]:
            fig_train_acc, ax_train_acc = plt.subplots()
            ax_train_acc.plot(train_scores, label='训练R²')
            ax_train_acc.set_xlabel('Epoch')
            ax_train_acc.set_ylabel('R²')
            ax_train_acc.set_title('训练集R²')
            st.pyplot(fig_train_acc)
        with row_acc[1]:
            fig_test_acc, ax_test_acc = plt.subplots()
            ax_test_acc.plot(test_scores, label='测试R²', color='orange')
            ax_test_acc.set_xlabel('Epoch')
            ax_test_acc.set_ylabel('R²')
            ax_test_acc.set_title('测试集R²')
            st.pyplot(fig_test_acc)

        # 损失和学习率曲线
        row_loss = st.columns(2)
        with row_loss[0]:
            st.subheader("训练损失曲线")
            fig_loss, ax_loss = plt.subplots()
            ax_loss.plot(losses)
            ax_loss.set_xlabel('Epoch')
            ax_loss.set_ylabel('Loss')
            st.pyplot(fig_loss)
        with row_loss[1]:
            st.subheader("学习率变化曲线")
            fig_lr, ax_lr = plt.subplots()
            ax_lr.plot(lrs)
            ax_lr.set_xlabel('Epoch')
            ax_lr.set_ylabel('Learning Rate')
            st.pyplot(fig_lr)

        # 评估
        model.eval()
        with torch.no_grad():
            if model_type == "MLP":
                X_test_eval = X_test_t.reshape(X_test_t.size(0), -1)
            else:
                X_test_eval = X_test_t
            y_pred = model(X_test_eval).cpu().numpy()
            y_true = y_test
        # 反归一化
        if y_pred.ndim == 1 or y_pred.shape[1] == 1:
            y_pred_inv = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).squeeze()
            y_true_inv = scaler_y.inverse_transform(y_true.reshape(-1, 1)).squeeze()
        else:
            y_pred_inv = scaler_y.inverse_transform(y_pred)
            y_true_inv = scaler_y.inverse_transform(y_true)
        # 主要指标
        if output_size == 1:
            mse = np.mean((y_true_inv - y_pred_inv) ** 2)
            mae = np.mean(np.abs(y_true_inv - y_pred_inv))
            r2 = 1 - np.sum((y_true_inv - y_pred_inv) ** 2) / np.sum((y_true_inv - np.mean(y_true_inv)) ** 2)
            st.write(f"MSE: {mse:.4f}  |  MAE: {mae:.4f}  |  R²: {r2:.4f}")
        else:
            metrics = []
            for i in range(output_size):
                mse = np.mean((y_true_inv[:, i] - y_pred_inv[:, i]) ** 2)
                mae = np.mean(np.abs(y_true_inv[:, i] - y_pred_inv[:, i]))
                r2 = 1 - np.sum((y_true_inv[:, i] - y_pred_inv[:, i]) ** 2) / np.sum((y_true_inv[:, i] - np.mean(y_true_inv[:, i])) ** 2)
                metrics.append([mse, mae, r2])
            metrics_df = pd.DataFrame(metrics, columns=["MSE", "MAE", "R²"], index=[f"step_{i+1}" for i in range(output_size)])
            st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)
        # 可视化
        st.subheader("测试集预测-真实值对比")
        if output_size == 1:
            fig_pred, ax_pred = plt.subplots()
            ax_pred.plot(time_test, y_true_inv, label='真实值', marker='o', color='blue')
            ax_pred.plot(time_test, y_pred_inv, label='预测值', marker='x', color='orange')
            ax_pred.set_xlabel('时间')
            ax_pred.set_ylabel('数值')
            ax_pred.legend()
            st.pyplot(fig_pred)
        else:
            for i in range(output_size):
                fig_pred, ax_pred = plt.subplots()
                ax_pred.plot(time_test, y_true_inv[:, i], label='真实值', marker='o', color='blue')
                ax_pred.plot(time_test, y_pred_inv[:, i], label='预测值', marker='x', color='orange')
                ax_pred.set_xlabel('时间')
                ax_pred.set_ylabel('数值')
                ax_pred.set_title(f'步长 step_{i+1}')
                ax_pred.legend()
                st.pyplot(fig_pred)
        # 模型导出
        st.subheader("模型导出")
        # .pt导出
        pt_filename = f"dl_ts_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        torch.save(model.state_dict(), pt_filename)
        with open(pt_filename, "rb") as f:
            st.download_button("下载 PyTorch .pt 模型", data=f, file_name=pt_filename)
        # .onnx导出（可选）
        if st.checkbox("导出为ONNX格式"):
            onnx_filename = pt_filename.replace('.pt', '.onnx')
            dummy_input = torch.randn(1, *X_test.shape[1:], dtype=torch.float32).to(device)
            torch.onnx.export(model, dummy_input, onnx_filename, input_names=['input'], output_names=['output'],
                              dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
                              opset_version=12)
            with open(onnx_filename, "rb") as f:
                st.download_button("下载 ONNX 模型", data=f, file_name=onnx_filename)

    # 10. 预留：动态导入新模型结构
    # 你可以在本目录下新建自定义模型类，并在模型选择区自动import并加入model_type列表。
    # 未来可用importlib动态加载自定义模型。
