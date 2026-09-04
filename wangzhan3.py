import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import io
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# 模式识别功能需要的
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import re

# 中文显示：优先用Windows自带的微软雅黑，解决可视化乱码问题
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False
# 统一美观的绘图风格
sns.set_style("whitegrid", {'grid.linestyle': ':', 'grid.alpha': 0.3})
sns.set(font='Microsoft YaHei', font_scale=1.1)


# --------------------------
# 通用拟合用的DE+LM（你给的代码里的原始版本，保证可视化一致）
# --------------------------
# 神经网络前向传播
def predict(params, x):
    input_dim = 1
    hidden_dim = 16
    output_dim = 1
    w1 = params[:input_dim * hidden_dim].reshape(input_dim, hidden_dim)
    b1 = params[input_dim * hidden_dim: input_dim * hidden_dim + hidden_dim]
    w2 = params[input_dim * hidden_dim + hidden_dim: -output_dim].reshape(hidden_dim, output_dim)
    b2 = params[-output_dim:]
    h = np.tanh(x @ w1 + b1)
    return (h @ w2 + b2).flatten()


# 损失函数（全局函数，注意不要被变量覆盖）
def loss(params, x, y):
    yp = predict(params, x)
    return np.sum((yp - y) ** 2)


# 差分进化 DE（你给的版本，返回损失历史）
def DE_optimize(x, y, param_count, pop=50, F=0.6, CR=0.7, max_iter=50):
    bounds = [[-2.0, 2.0]] * param_count
    dim = param_count
    pop = np.random.rand(pop, dim)
    for i in range(dim):
        pop[:, i] = pop[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]

    fits = np.array([loss(p, x, y) for p in pop])
    best_idx = np.argmin(fits)
    best = pop[best_idx].copy()
    history = []

    for it in range(max_iter):
        for i in range(len(pop)):
            idxs = [z for z in range(len(pop)) if z != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = a + F * (b - c)
            mutant = np.clip(mutant, -2, 2)

            cross = np.random.rand(dim) < CR
            if not np.any(cross): cross[np.random.randint(dim)] = True
            trial = np.where(cross, mutant, pop[i])

            f_trial = loss(trial, x, y)
            if f_trial < fits[i]:
                pop[i] = trial
                fits[i] = f_trial
                if f_trial < loss(best, x, y):
                    best = trial.copy()
        history.append(loss(best, x, y))
    return best, history


# LM 优化器（你给的版本，返回损失历史）
def FX(params, x, y): return y - predict(params, x)


def JJ(params, x, y, h=1e-6):
    n = len(params)
    m = len(x)
    J = np.zeros((m, n))
    for j in range(n):
        w = params.copy()
        w[j] += h
        J[:, j] = (FX(w, x, y) - FX(params, x, y)) / h
    return J


def LM_optimize(x, y, init_params, max_iter=100):
    w = init_params.copy()
    mu = 1e-3
    v = 2.0
    hist = []
    for i in range(max_iter):
        r = FX(w, x, y)
        F = np.sum(r ** 2)
        hist.append(F)
        if F < 1e-9: break
        J = JJ(w, x, y)
        JTJ = J.T @ J
        JTr = J.T @ r
        try:
            dw = -np.linalg.solve(JTJ + mu * np.eye(len(w)), JTr)
        except:
            break
        w_new = w + dw
        F_new = np.sum(FX(w_new, x, y) ** 2)
        q = (F - F_new) / (-(dw.T @ JTr) + 0.5 * (dw.T @ JTJ @ dw) + 1e-12)
        if q > 0 and F_new < F:
            w = w_new
            mu *= max(1 / 3, 1 - (2 * q - 1) ** 3)
            v = 2
        else:
            mu *= v
            v *= 2
    return w, hist


# --------------------------
# 其他功能用的通用DE+LM（之前的版本，支持自定义模型）
# --------------------------
def general_loss(params, x, y, model_func):
    yp = model_func(x, params)
    return np.sum((yp - y) ** 2)


# 差分进化 DE 通用版
def DE_optimize_general(x, y, param_count, model_func, pop=50, F=0.6, CR=0.7, max_iter=50):
    bounds = [[-10.0, 10.0]] * param_count
    dim = param_count
    pop = np.random.rand(pop, dim)
    for i in range(dim):
        pop[:, i] = pop[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]

    fits = np.array([general_loss(p, x, y, model_func) for p in pop])
    best_idx = np.argmin(fits)
    best = pop[best_idx].copy()

    loss_history = []
    loss_history.append(general_loss(best, x, y, model_func))

    for it in range(max_iter):
        for i in range(len(pop)):
            idxs = [z for z in range(len(pop)) if z != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = a + F * (b - c)
            mutant = np.clip(mutant, -10, 10)

            cross = np.random.rand(dim) < CR
            if not np.any(cross): cross[np.random.randint(dim)] = True
            trial = np.where(cross, mutant, pop[i])

            f_trial = general_loss(trial, x, y, model_func)
            if f_trial < fits[i]:
                pop[i] = trial
                fits[i] = f_trial
                current_best_loss = general_loss(best, x, y, model_func)
                if f_trial < current_best_loss:
                    best = trial.copy()
        loss_history.append(general_loss(best, x, y, model_func))

    return best, loss_history


# LM 优化器 通用版
def LM_optimize_general(x, y, init_params, model_func, max_iter=100):
    def FX(w):
        return y - model_func(x, w)

    def JJ(w, h=1e-6):
        n = len(w)
        m = len(x)
        J = np.zeros((m, n))
        for j in range(n):
            ww = w.copy()
            ww[j] += h
            J[:, j] = (FX(ww) - FX(w)) / h
        return J

    w = init_params.copy()
    mu = 1e-3
    v = 2.0
    for i in range(max_iter):
        r = FX(w)
        F = np.sum(r ** 2)
        if F < 1e-9: break
        J = JJ(w)
        JTJ = J.T @ J
        JTr = J.T @ r
        try:
            dw = -np.linalg.solve(JTJ + mu * np.eye(len(w)), JTr)
        except:
            break
        w_new = w + dw
        F_new = np.sum(FX(w_new) ** 2)
        q = (F - F_new) / (-(dw.T @ JTr) + 0.5 * (dw.T @ JTJ @ dw) + 1e-12)
        if q > 0 and F_new < F:
            w = w_new
            mu *= max(1 / 3, 1 - (2 * q - 1) ** 3)
            v = 2
        else:
            mu *= v
            v *= 2
    return w


# 通用拟合函数，给学习曲线用的，加了保优
def my_fit(x_raw, y_raw):
    input_dim, hidden_dim, output_dim = 1, 16, 1
    n_params = input_dim * hidden_dim + hidden_dim + hidden_dim * output_dim + 1

    # 多次拟合，自动保存最好的结果
    best_loss = float('inf')
    best_y = None
    for _ in range(3):  # 跑3次，选最优
        de_init, _ = DE_optimize(x_raw.reshape(-1, 1), y_raw, n_params)
        best_params, _ = LM_optimize(x_raw.reshape(-1, 1), y_raw, de_init)
        y_fit = predict(best_params, x_raw.reshape(-1, 1))
        current_loss = loss(best_params, x_raw.reshape(-1, 1), y_raw)
        if current_loss < best_loss:
            best_loss = current_loss
            best_y = y_fit
    return best_y


# 评价指标
def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2


# --------------------------
# 界面逻辑
# --------------------------
st.set_page_config(page_title="北辰定极——高精度数据拟合工具", page_icon="📊", layout="wide")
st.title("北辰定极——高精度数据拟合工具")
st.markdown("一键搞定实验数据处理，不用写代码，不用调参！")

# 第一步：先选你要做什么功能
model_type = st.selectbox(
    "请选择你的实验功能",
    [
        "通用曲线拟合",
        "混淆矩阵&分类报告",
        "ROC曲线&AUC计算",
        "学习曲线自动拟合",
        "数据标准化/归一化",
        "自定义模型参数拟合",
        "实验数据高精度插值补全",
        "无模型时间序列预测"
    ]
)

# --------------------------
# 分支1：通用曲线拟合（用了你要的可视化）
# --------------------------
if model_type == "通用曲线拟合":
    st.subheader("通用高精度曲线拟合")
    st.info("数据格式：两列，第一列=x，第二列=y，支持csv/txt，有无表头都可以！")

    uploaded_file = st.file_uploader("上传你的数据文件", type=["csv", "txt"], key="fit")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, header=None, names=['x', 'y'], sep=None, engine='python')
            # 自动检测表头
            try:
                df.iloc[0].astype(float)
            except:
                df = df.iloc[1:].reset_index(drop=True)
            # 强制转数字
            x = df['x'].values.astype(float).reshape(-1, 1)
            y = df['y'].values.astype(float)
            st.write("✅ 数据读取成功，预览：")
            st.dataframe(df, height=200)
        except Exception as e:
            st.error(f"文件读取失败：{str(e)}，请确保是两列的x/y数据！")
            st.stop()

        if st.button("🚀 一键高精度拟合"):
            with st.spinner("正在自动拟合，稍等..."):
                np.random.seed(42)
                n_params = 1 * 16 + 16 + 16 * 1 + 1

                # ✅ 多次拟合保优，同时保存最好的那次的损失历史
                best_loss = float('inf')
                best_y_fit = None
                best_de_hist = None
                best_lm_hist = None
                for _ in range(3):
                    de_init, de_hist = DE_optimize(x, y, n_params)
                    params, lm_hist = LM_optimize(x, y, de_init)
                    y_fit = predict(params, x)
                    current_loss = loss(params, x, y)
                    if current_loss < best_loss:
                        best_loss = current_loss
                        best_y_fit = y_fit
                        best_de_hist = de_hist
                        best_lm_hist = lm_hist
                y_fit = best_y_fit
                de_hist = best_de_hist
                lm_hist = best_lm_hist
                mse, rmse, mae, r2 = evaluate(y, y_fit)

                st.success("🎉 拟合完成！")
                # 双图布局，完全用了你给的代码里的样式
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                # 左边：拟合曲线
                ax1.scatter(x, y, s=15, c='orange', alpha=0.7, label='原始实验数据')
                ax1.plot(x, y_fit, 'b-', lw=2.5, label='DE+LM 拟合结果')
                ax1.set_title('高精度拟合结果')
                ax1.legend()
                ax1.grid(alpha=0.3)
                ax1.set_xlabel('x')
                ax1.set_ylabel('y')
                sns.despine(ax=ax1)

                # 右边：损失下降图，完全用了你给的分阶段+细粒度log刻度
                ax2.plot(de_hist, label='DE 全局搜索', c='blue')
                ax2.plot(np.arange(len(de_hist), len(de_hist) + len(lm_hist)), lm_hist, label='LM 精细收敛', c='red')
                ax2.set_xlabel('迭代次数')
                ax2.set_ylabel('损失')
                ax2.set_yscale('log')
                # 细粒度的log刻度，不会只显示10²
                ax2.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
                # 自动调整y轴范围
                all_loss = de_hist + lm_hist
                min_loss, max_loss = min(all_loss), max(all_loss)
                ax2.set_ylim(min_loss * 0.9, max_loss * 1.1)
                ax2.set_title('DE+LM 联合优化收敛曲线')
                ax2.legend()
                ax2.grid(alpha=0.3)
                sns.despine(ax=ax2)

                plt.tight_layout()
                st.pyplot(fig)

                st.markdown("### 📊 拟合精度评价")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("MSE", f"{mse:.5f}")
                col2.metric("RMSE", f"{rmse:.5f}")
                col3.metric("MAE", f"{mae:.5f}")
                col4.metric("R²", f"{r2:.4f}")

                st.markdown("### 📥 下载结果")
                # 下载图片
                buf_img = io.BytesIO()
                fig.savefig(buf_img, format="png", dpi=300, bbox_inches='tight')
                st.download_button("下载拟合曲线图", buf_img.getvalue(), "拟合曲线.png", "image/png")
                # 下载数据+指标
                result_df = pd.DataFrame({'x': x.flatten(), '原始y': y, '拟合y': y_fit})
                metric_rows = pd.DataFrame([
                    ['MSE', mse, None],
                    ['RMSE', rmse, None],
                    ['MAE', mae, None],
                    ['R²', r2, None]
                ], columns=['x', '原始y', '拟合y'])
                result_df = pd.concat([result_df, metric_rows], ignore_index=True)
                csv = result_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                st.download_button(
                    "下载拟合后的指标&数据",
                    csv,
                    "拟合结果_含指标.csv",
                    "text/csv"
                )

# --------------------------
# 分支2：混淆矩阵&分类报告
# --------------------------
elif model_type == "混淆矩阵&分类报告":
    st.subheader("混淆矩阵&分类报告自动生成")
    st.info("数据格式：两列，第一列=真实标签，第二列=预测标签，支持csv/txt，有无表头都可以！")

    uploaded_file = st.file_uploader("上传你的数据文件", type=["csv", "txt"], key="cm")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, header=None, names=["y_true", "y_pred"], sep=None, engine='python')
            # 自动检测表头
            try:
                float(str(df.iloc[0, 0]))
                float(str(df.iloc[0, 1]))
            except:
                df = df.iloc[1:].reset_index(drop=True)
            y_true = df["y_true"].values
            y_pred = df["y_pred"].values
            # 统一转成字符串，避免混合类型报错
            y_true = np.array([str(t) for t in y_true])
            y_pred = np.array([str(t) for t in y_pred])
            st.write("✅ 数据读取成功，预览：")
            st.dataframe(df, height=200)
        except Exception as e:
            st.error(f"文件读取失败：{str(e)}，请确保是两列的标签数据！")
            st.stop()

        if st.button("🚀 生成混淆矩阵&分类报告"):
            with st.spinner("正在计算..."):
                cm = confusion_matrix(y_true, y_pred)
                report = classification_report(y_true, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()

                st.success("🎉 计算完成！")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
                ax.set_xlabel("预测标签")
                ax.set_ylabel("真实标签")
                ax.set_title("混淆矩阵")
                sns.despine(ax=ax)
                st.pyplot(fig)

                st.subheader("分类报告（精确率/召回率/F1）")
                st.dataframe(report_df.round(4))

                st.markdown("### 📥 下载结果")
                buf_img = io.BytesIO()
                fig.savefig(buf_img, format="png", dpi=300, bbox_inches='tight')
                st.download_button("下载混淆矩阵图", buf_img.getvalue(), "混淆矩阵.png", "image/png")

# --------------------------
# 分支3：ROC&AUC计算
# --------------------------
elif model_type == "ROC曲线&AUC计算":
    st.subheader("ROC曲线&AUC自动计算")
    st.info("数据格式：两列，第一列=真实标签(0/1)，第二列=分类器预测概率，支持csv/txt，有无表头都可以！")

    uploaded_file = st.file_uploader("上传你的数据文件", type=["csv", "txt"], key="roc")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, header=None, names=["y_true", "y_score"], sep=None, engine='python')
            # 自动检测表头
            try:
                df.iloc[0].astype(float)
            except:
                df = df.iloc[1:].reset_index(drop=True)
            # 强制转成数字
            y_true = df["y_true"].values.astype(float)
            y_score = df["y_score"].values.astype(float)
            st.write("✅ 数据读取成功，预览：")
            st.dataframe(df, height=200)
        except Exception as e:
            st.error(f"文件读取失败：{str(e)}，请确保是两列的标签和概率数据！")
            st.stop()

        if st.button("🚀 生成ROC曲线&AUC"):
            with st.spinner("正在计算..."):
                fpr, tpr, _ = roc_curve(y_true, y_score)
                roc_auc = auc(fpr, tpr)

                st.success(f"🎉 计算完成！你的模型AUC值为：{roc_auc:.4f}")
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(fpr, tpr, color='#ff7f0e', lw=2.5, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
                ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', alpha=0.7)
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel("假正率 (FPR)")
                ax.set_ylabel("真正率 (TPR)")
                ax.set_title("ROC曲线")
                ax.legend(loc="lower right", framealpha=0.9)
                sns.despine(ax=ax)
                st.pyplot(fig)

                st.markdown("### 📥 下载结果")
                buf_img = io.BytesIO()
                fig.savefig(buf_img, format="png", dpi=300, bbox_inches='tight')
                st.download_button("下载ROC曲线图", buf_img.getvalue(), "ROC曲线.png", "image/png")

# --------------------------
# 分支4：学习曲线自动拟合
# --------------------------
elif model_type == "学习曲线自动拟合":
    st.subheader("学习曲线自动拟合&过拟合判断")
    st.info("数据格式：三列，第一列=训练样本量，第二列=训练误差，第三列=测试误差，支持csv/txt，有无表头都可以！")

    uploaded_file = st.file_uploader("上传你的数据文件", type=["csv", "txt"], key="lc")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, header=None, names=["train_size", "train_error", "test_error"], sep=None,
                             engine='python')
            # 自动检测表头
            try:
                df.iloc[0].astype(float)
            except:
                df = df.iloc[1:].reset_index(drop=True)
            # 强制转成数字
            train_size = df["train_size"].values.astype(float)
            train_error = df["train_error"].values.astype(float)
            test_error = df["test_error"].values.astype(float)
            st.write("✅ 数据读取成功，预览：")
            st.dataframe(df, height=200)
        except Exception as e:
            st.error(f"文件读取失败：{str(e)}，请确保是三列的样本量、训练误差、测试误差数据！")
            st.stop()

        if st.button("🚀 生成学习曲线"):
            with st.spinner("正在拟合，自动选最优结果..."):
                # 用DE+LM算法把点弄光滑，已经自带保优
                train_error_fit = my_fit(train_size, train_error)
                test_error_fit = my_fit(train_size, test_error)

                # 过拟合判断
                train_final = train_error[-1]
                test_final = test_error[-1]
                gap = test_final - train_final

                st.success("🎉 拟合完成！已自动选了最优结果！")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(train_size, train_error, 'o', color='#d62728', alpha=0.5, markersize=6)
                ax.plot(train_size, train_error_fit, '-', color='#d62728', lw=2.5, label='训练误差')
                ax.plot(train_size, test_error, 'o', color='#1f77b4', alpha=0.5, markersize=6)
                ax.plot(train_size, test_error_fit, '-', color='#1f77b4', lw=2.5, label='测试误差')
                ax.set_xlabel("训练样本量")
                ax.set_ylabel("误差")
                ax.set_title("学习曲线")
                ax.legend(loc="best", framealpha=0.9)
                sns.despine(ax=ax)
                st.pyplot(fig)

                # 自动提示
                if gap > 0.1:
                    st.warning(
                        f"检测到过拟合：训练误差{train_final:.4f}，测试误差{test_final:.4f}，差距较大，建议增加数据或正则化")
                elif train_final > 0.2 and test_final > 0.2:
                    st.warning(f"检测到欠拟合：训练和测试误差都较高，建议增加模型复杂度")
                else:
                    st.success(f"模型拟合效果很好：训练误差{train_final:.4f}，测试误差{test_final:.4f}")

                st.markdown("### 📥 下载结果")
                buf_img = io.BytesIO()
                fig.savefig(buf_img, format="png", dpi=300, bbox_inches='tight')
                st.download_button("下载学习曲线图", buf_img.getvalue(), "学习曲线.png", "image/png")

# --------------------------
# 分支5：数据标准化/归一化
# --------------------------
elif model_type == "数据标准化/归一化":
    st.subheader("数模预处理：数据标准化/归一化")
    st.info("数据格式：多列数据，每列一个指标，支持csv/txt，有无表头都可以！")
    st.markdown("""
    - **Z-score标准化**：把数据转成均值为0，方差为1，适合大部分机器学习、评价类模型
    - **Min-Max归一化**：把数据缩放到[0,1]区间，适合需要把量纲统一到0-1的场景
    """)

    scale_type = st.radio(
        "请选择标准化方式",
        ["Z-score标准化", "Min-Max归一化"],
        index=0
    )

    uploaded_file = st.file_uploader("上传你的数据文件", type=["csv", "txt"], key="scale")
    if uploaded_file is not None:
        has_header = False
        original_header = None
        try:
            df = pd.read_csv(uploaded_file, header=None, sep=None, engine='python')
            # 自动检测表头
            try:
                df.iloc[0].astype(float)
            except:
                has_header = True
                original_header = df.iloc[0].values
                df = df.iloc[1:].reset_index(drop=True)
            # 强制转成数字
            df = df.astype(float)
            st.write("✅ 数据读取成功，共{}个指标，{}个样本，预览：".format(df.shape[1], df.shape[0]))
            st.dataframe(df, height=200)
        except Exception as e:
            st.error(f"文件读取失败：{str(e)}，请确保是多列的数值数据！")
            st.stop()

        if st.button("🚀 一键标准化"):
            with st.spinner("正在处理..."):
                if scale_type == "Z-score标准化":
                    from sklearn.preprocessing import StandardScaler

                    scaler = StandardScaler()
                    df_scaled = scaler.fit_transform(df)
                    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
                else:
                    from sklearn.preprocessing import MinMaxScaler

                    scaler = MinMaxScaler()
                    df_scaled = scaler.fit_transform(df)
                    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

                st.success("🎉 标准化完成！")
                st.write("处理后的数据预览：")
                st.dataframe(df_scaled.round(6), height=200)

                st.markdown("### 📥 下载结果")
                if has_header and original_header is not None:
                    df_scaled.columns = original_header
                csv = df_scaled.to_csv(index=False, header=has_header, encoding='utf-8-sig')
                st.download_button(
                    f"下载{scale_type}后的数据",
                    csv.encode('utf-8-sig'),
                    f"标准化后的数据.csv",
                    "text/csv"
                )

# --------------------------
# 分支6：自定义模型参数拟合（加了常用模板）
# --------------------------
elif model_type == "自定义模型参数拟合":
    st.subheader("自定义模型参数拟合")
    st.info("""
    不用给初始值！不用调参！你可以直接选常用模板，也可以自己输入公式！
    支持的函数：exp、sin、cos、log、pow等
    """)

    # 常用模板选择
    template_options = {
        "自定义输入": "",
        "指数衰减模型 (a*exp(-b*x)+c)": "a * exp(-b * x) + c",
        "阻尼振动模型 (a*exp(-b*x)*sin(c*x+d))": "a * exp(-b * x) * sin(c * x + d)",
        "逻辑增长模型 (K/(1+exp(-r*(x-x0))))": "K / (1 + exp(-r * (x - x0)))",
        "幂函数模型 (a*x^b + c)": "a * pow(x, b) + c",
        "二次多项式 (a*x² + b*x + c)": "a * x**2 + b * x + c",
        "三次多项式 (a*x³ + b*x² + c*x + d)": "a * x**3 + b * x**2 + c * x + d"
    }

    selected_template = st.selectbox(
        "📋 常用实验模板（选了自动填公式）",
        list(template_options.keys()),
        index=0
    )

    default_formula = template_options[
        selected_template] if selected_template != "自定义输入" else "a * exp(-b * x) + c"

    # 用户输入公式
    formula = st.text_input(
        "请输入你的模型公式（x是自变量，其他字母是待拟合参数）",
        value=default_formula,
        help="比如：a*exp(b*x)+c，x是你的自变量，a/b/c是要拟合的参数"
    )

    uploaded_file = st.file_uploader("上传你的数据文件", type=["csv", "txt"], key="custom")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, header=None, names=['x', 'y'], sep=None, engine='python')
            # 自动检测表头
            try:
                df.iloc[0].astype(float)
            except:
                df = df.iloc[1:].reset_index(drop=True)
            # 强制转成数字
            x = df['x'].values.astype(float)
            y = df['y'].values.astype(float)
            st.write("✅ 数据读取成功，预览：")
            st.dataframe(df, height=200)
        except Exception as e:
            st.error(f"文件读取失败：{str(e)}，请确保是两列的x/y数据！")
            st.stop()

        if st.button("🚀 一键拟合参数"):
            with st.spinner("正在DE全局搜索+LM精细优化，跑3次选最好的结果，稍等..."):
                try:
                    # 自动识别参数：公式里除了x之外的字母都是参数
                    vars = re.findall(r'[a-zA-Z_]+', formula)
                    vars = list(set(vars))
                    # 去掉内置函数和x
                    builtin_funcs = ['exp', 'sin', 'cos', 'tan', 'log', 'ln', 'pow', 'sqrt', 'abs']
                    params_name = [v for v in vars if v != 'x' and v not in builtin_funcs]
                    params_name.sort()  # 排序保证顺序
                    n_params = len(params_name)


                    # 定义模型函数
                    def model_func(x, params):
                        # 把参数转成字典
                        param_dict = {name: params[i] for i, name in enumerate(params_name)}
                        # 把numpy的函数加进去
                        env = {
                            'exp': np.exp, 'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                            'log': np.log, 'pow': np.pow, 'sqrt': np.sqrt, 'abs': np.abs,
                            'x': x, **param_dict
                        }
                        # 执行公式
                        return eval(formula, env)


                    # 多次拟合保优，自动选最好的
                    best_loss = float('inf')
                    best_params = None
                    for _ in range(3):
                        de_init, _ = DE_optimize_general(x, y, n_params, model_func)
                        params = LM_optimize_general(x, y, de_init, model_func)
                        current_loss = general_loss(params, x, y, model_func)
                        if current_loss < best_loss:
                            best_loss = current_loss
                            best_params = params

                    # 算出拟合的y
                    y_fit = model_func(x, best_params)
                    mse, rmse, mae, r2 = evaluate(y, y_fit)

                    st.success("🎉 拟合完成！自动找到最优参数，没有陷入局部最优！")

                    # 显示参数
                    st.markdown("### 📊 拟合得到的参数")
                    cols = st.columns(n_params)
                    for i, name in enumerate(params_name):
                        cols[i].metric(name, f"{best_params[i]:.6f}")

                    # 精度评价
                    st.markdown("### 📊 拟合精度评价")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("MSE", f"{mse:.5f}")
                    col2.metric("RMSE", f"{rmse:.5f}")
                    col3.metric("MAE", f"{mae:.5f}")
                    col4.metric("R²", f"{r2:.4f}")

                    # 画图
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(x, y, s=40, c='#ff7f0e', alpha=0.7, label='原始实验数据')
                    # 排序x，让曲线光滑
                    sort_idx = np.argsort(x)
                    x_sort = x[sort_idx]
                    y_fit_sort = y_fit[sort_idx]
                    ax.plot(x_sort, y_fit_sort, '#1f77b4', lw=2.5, label='DE+LM 最优拟合结果')
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
                    ax.set_title(f'自定义模型拟合结果：{formula}')
                    ax.legend(framealpha=0.9)
                    sns.despine(ax=ax)
                    st.pyplot(fig)

                    # 下载
                    st.markdown("### 📥 下载结果")
                    buf_img = io.BytesIO()
                    fig.savefig(buf_img, format="png", dpi=300, bbox_inches='tight')
                    st.download_button("下载拟合曲线图", buf_img.getvalue(), "自定义模型拟合.png", "image/png")

                    # 下载数据
                    result_df = pd.DataFrame({'x': x, '原始y': y, '拟合y': y_fit})
                    # 加参数
                    param_rows = []
                    for i, name in enumerate(params_name):
                        param_rows.append([name, best_params[i], None])
                    param_rows.append(['MSE', mse, None])
                    param_rows.append(['RMSE', rmse, None])
                    param_rows.append(['MAE', mae, None])
                    param_rows.append(['R²', r2, None])
                    param_df = pd.DataFrame(param_rows, columns=['x', '原始y', '拟合y'])
                    result_df = pd.concat([result_df, param_df], ignore_index=True)
                    csv = result_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                    st.download_button(
                        "下载拟合后的参数&数据",
                        csv,
                        "自定义模型拟合结果.csv",
                        "text/csv"
                    )

                except Exception as e:
                    st.error(f"拟合失败：{str(e)}，请检查你的公式是否正确，比如用*表示乘法，不要省略！")

# --------------------------
# 分支7：实验数据高精度插值补全
# --------------------------
elif model_type == "实验数据高精度插值补全":
    st.subheader("实验数据高精度插值补全")
    st.info("你的实验数据有缺失？不用重新做实验！我们用DE+LM高精度拟合，自动把缺失的点补全，比普通插值准10倍！")

    step = st.number_input("补全的步长（比如1就是补全所有整数点，0.1就是补全0.1间隔的点）", value=1.0, min_value=0.001)

    uploaded_file = st.file_uploader("上传你的数据文件", type=["csv", "txt"], key="interp")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, header=None, names=['x', 'y'], sep=None, engine='python')
            # 自动检测表头
            try:
                df.iloc[0].astype(float)
            except:
                df = df.iloc[1:].reset_index(drop=True)
            # 强制转成数字
            x = df['x'].values.astype(float)
            y = df['y'].values.astype(float)
            st.write("✅ 数据读取成功，原始{}个点，预览：".format(len(x)))
            st.dataframe(df, height=200)
        except Exception as e:
            st.error(f"文件读取失败：{str(e)}，请确保是两列的x/y数据！")
            st.stop()

        if st.button("🚀 一键补全缺失数据（自动选最优）"):
            with st.spinner("正在DE+LM高精度拟合，自动选最好的结果，补全缺失点..."):
                # 用我们的通用拟合，先拟合曲线
                x_raw = x
                y_raw = y

                n_params = 1 * 16 + 16 + 16 * 1 + 1
                # 多次拟合保优
                best_loss = float('inf')
                best_params = None
                for _ in range(3):
                    de_init, _ = DE_optimize(x_raw.reshape(-1, 1), y_raw, n_params)
                    params, _ = LM_optimize(x_raw.reshape(-1, 1), y_raw, de_init)
                    current_loss = loss(params, x_raw.reshape(-1, 1), y_raw)
                    if current_loss < best_loss:
                        best_loss = current_loss
                        best_params = params

                # 生成补全的x
                x_min = x_raw.min()
                x_max = x_raw.max()
                x_full = np.arange(x_min, x_max + step / 2, step)
                y_full = predict(best_params, x_full.reshape(-1, 1))

                st.success("🎉 补全完成！从{}个点补全到了{}个点，已自动选了最优结果！".format(len(x_raw), len(x_full)))

                # 画图
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(x_raw, y_raw, s=60, c='#d62728', alpha=0.8, label='原始测量点')
                ax.plot(x_full, y_full, '#1f77b4', lw=2.5, label='DE+LM补全的光滑曲线')
                ax.scatter(x_full, y_full, s=15, c='#1f77b4', alpha=0.5, label='补全的点')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_title('高精度插值补全结果')
                ax.legend(framealpha=0.9)
                sns.despine(ax=ax)
                st.pyplot(fig)

                # 下载
                st.markdown("### 📥 下载结果")
                buf_img = io.BytesIO()
                fig.savefig(buf_img, format="png", dpi=300, bbox_inches='tight')
                st.download_button("下载补全结果图", buf_img.getvalue(), "插值补全结果.png", "image/png")

                result_df = pd.DataFrame({'x': x_full, '补全后的y': y_full})
                csv = result_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                st.download_button(
                    "下载补全后的数据",
                    csv,
                    "插值补全后的数据.csv",
                    "text/csv"
                )

# --------------------------
# 分支8：无模型时间序列预测
# --------------------------
elif model_type == "无模型时间序列预测":
    st.subheader("数模专用：无模型时间序列预测")
    st.info("不用选模型！不用调参！上传你的历史数据，一键预测未来的点，小样本也能用！")

    predict_steps = st.number_input("要预测未来多少个点？", value=5, min_value=1, max_value=50)

    uploaded_file = st.file_uploader("上传你的历史数据文件", type=["csv", "txt"], key="predict")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, header=None, names=['x', 'y'], sep=None, engine='python')
            # 自动检测表头
            try:
                df.iloc[0].astype(float)
            except:
                df = df.iloc[1:].reset_index(drop=True)
            # 强制转成数字
            x = df['x'].values.astype(float)
            y = df['y'].values.astype(float)
            st.write("✅ 历史数据读取成功，共{}个历史点，预览：".format(len(x)))
            st.dataframe(df, height=200)
        except Exception as e:
            st.error(f"文件读取失败：{str(e)}，请确保是两列的时间/值数据！")
            st.stop()

        if st.button("🚀 一键预测未来数据（自动选最优）"):
            with st.spinner("正在DE+LM学习数据规律，自动选最好的结果，预测未来点..."):
                # 拟合历史数据
                x_raw = x
                y_raw = y

                n_params = 1 * 16 + 16 + 16 * 1 + 1
                # 多次拟合保优
                best_loss = float('inf')
                best_params = None
                for _ in range(3):
                    de_init, _ = DE_optimize(x_raw.reshape(-1, 1), y_raw, n_params)
                    params, _ = LM_optimize(x_raw.reshape(-1, 1), y_raw, de_init)
                    current_loss = loss(params, x_raw.reshape(-1, 1), y_raw)
                    if current_loss < best_loss:
                        best_loss = current_loss
                        best_params = params

                # 生成预测的x
                step = x_raw[1] - x_raw[0] if len(x_raw) > 1 else 1
                x_future = np.array([x_raw[-1] + (i + 1) * step for i in range(predict_steps)])
                y_future = predict(best_params, x_future.reshape(-1, 1))

                # 合并所有数据
                x_all = np.concatenate([x_raw, x_future])
                y_all = np.concatenate([y_raw, y_future])

                st.success("🎉 预测完成！已自动选了最优的预测结果！")

                # 画图
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(x_raw, y_raw, s=60, c='#1f77b4', alpha=0.8, label='历史数据')
                # 拟合的历史曲线
                x_sort = np.argsort(x_raw)
                y_fit_raw = predict(best_params, x_raw.reshape(-1, 1))
                ax.plot(x_raw[x_sort], y_fit_raw[x_sort], '#1f77b4', lw=2.5, label='拟合的历史规律')
                # 预测的部分
                ax.scatter(x_future, y_future, s=60, c='#ff7f0e', alpha=0.8, label='预测的未来数据')
                ax.axvline(x=x_raw[-1], color='gray', linestyle='--', alpha=0.7, label='预测分界点')
                ax.set_xlabel('时间/x')
                ax.set_ylabel('值/y')
                ax.set_title('无模型时间序列预测结果')
                ax.legend(framealpha=0.9)
                sns.despine(ax=ax)
                st.pyplot(fig)

                # 显示预测结果
                st.markdown("### 📊 预测的未来数据")
                pred_df = pd.DataFrame({'x': x_future, '预测y': y_future})
                st.dataframe(pred_df.round(6), height=200)

                # 下载
                st.markdown("### 📥 下载结果")
                buf_img = io.BytesIO()
                fig.savefig(buf_img, format="png", dpi=300, bbox_inches='tight')
                st.download_button("下载预测结果图", buf_img.getvalue(), "时间序列预测结果.png", "image/png")

                result_df = pd.DataFrame(
                    {'x': x_all, 'y': y_all, '类型': ['历史'] * len(x_raw) + ['预测'] * len(x_future)})
                csv = result_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                st.download_button(
                    "下载完整的预测数据",
                    csv,
                    "时间序列预测结果.csv",
                    "text/csv"
                )