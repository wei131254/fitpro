import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib

# 加载你自己的中文字体
font_path = './SimHei.ttf'
font_prop = matplotlib.font_manager.FontProperties(fname=font_path)

# 然后你画图的时候，所有的中文都指定这个字体：
plt.title('DE+LM 算法拟合结果图', fontproperties=font_prop)
plt.xlabel('x轴', fontproperties=font_prop)
plt.ylabel('y轴', fontproperties=font_prop)
# 图例也要加
plt.legend(prop=font_prop)
# 中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# --------------------------
# 你的DE+LM算法，我完整的搬过来了
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


# 损失函数
def loss(params, x, y):
    yp = predict(params, x)
    return np.sum((yp - y) ** 2)


# 差分进化 DE
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


# LM 优化器
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
        dw = -np.linalg.solve(JTJ + mu * np.eye(len(w)), JTr)
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


# 评价指标
def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2


# --------------------------
# Streamlit 界面
# --------------------------
st.set_page_config(page_title="北辰定极——高精度数据拟合工具", page_icon="📊")
st.title("北辰定极——高精度数据拟合工具")
st.markdown("上传你的实验数据，一键完成高精度非线性拟合，无需调参，无需写代码！")
st.info("数据格式要求：两列数据，第一列是x，第二列是y，支持csv/txt格式，逗号/空格分隔都可以")

# 1. 上传文件
uploaded_file = st.file_uploader("上传你的数据文件", type=["csv", "txt"])

if uploaded_file is not None:
    # 自动读取数据，兼容各种分隔符
    try:
        # 先尝试读逗号分隔，不行就读空格分隔
        df = pd.read_csv(uploaded_file, header=None, names=['x', 'y'], sep=None, engine='python')
        x = df['x'].values.reshape(-1, 1)
        y = df['y'].values
        st.write("✅ 数据读取成功，预览：")
        st.dataframe(df, height=200)
    except Exception as e:
        st.error(f"文件读取失败：{str(e)}，请确保是两列的x/y数据！")
        st.stop()

    # 2. 拟合按钮
    if st.button("🚀 一键高精度拟合（无需调参）"):
        with st.spinner("正在自动拟合中，DE全局搜索+LM精细优化，稍等..."):
            np.random.seed(42)  # 固定种子，保证结果稳定

            # 计算参数数量
            input_dim, hidden_dim, output_dim = 1, 16, 1
            n_params = input_dim * hidden_dim + hidden_dim + hidden_dim * output_dim + output_dim

            # 你的DE+LM算法，直接跑
            st.write("1. 差分进化全局搜索初始点...")
            de_init, de_hist = DE_optimize(x, y, n_params)

            st.write("2. LM算法精细收敛...")
            best_params, lm_hist = LM_optimize(x, y, de_init)

            # 得到拟合结果
            y_fit = predict(best_params, x)
            mse, rmse, mae, r2 = evaluate(y, y_fit)

            # 3. 画图
            st.success("🎉 拟合完成！")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # 拟合结果图
            ax1.scatter(x, y, s=15, c='orange', alpha=0.7, label='原始实验数据')
            ax1.plot(x, y_fit, 'b-', lw=2.5, label='DE+LM 拟合结果')
            ax1.set_title('高精度拟合结果')
            ax1.legend()
            ax1.grid(alpha=0.3)

            # 收敛曲线
            ax2.plot(de_hist, label='DE 全局搜索', c='blue')
            ax2.plot(np.arange(len(de_hist), len(de_hist) + len(lm_hist)), lm_hist, label='LM 精细收敛', c='red')
            ax2.set_xlabel('迭代次数')
            ax2.set_ylabel('损失')
            ax2.set_yscale('log')
            ax2.set_title('DE+LM 联合优化收敛曲线')
            ax2.legend()
            ax2.grid(alpha=0.3)

            st.pyplot(fig)

            # 4. 输出评价指标
            st.markdown("### 📊 拟合精度评价")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MSE", f"{mse:.5f}")
            col2.metric("RMSE", f"{rmse:.5f}")
            col3.metric("MAE", f"{mae:.5f}")
            col4.metric("R²", f"{r2:.4f}")

            # 5. 下载结果
            st.markdown("### 📥 下载结果")
            result_df = pd.DataFrame({'x': x.flatten(), '原始y': y, '拟合y': y_fit})
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="下载拟合后的数据",
                data=csv,
                file_name='fitpro_result.csv',
                mime='text/csv',
            )
