import NewareNDA
import matplotlib.pyplot as plt
import os

# =============================
# 1. 画图风格（别折腾太花）
# =============================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.linewidth': 1.0
})

def apply_style(ax):
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_linewidth(1.0)

    ax.tick_params(axis='both',
                   direction='in',
                   length=6,
                   width=1.5,
                   labelsize=12) 
# =============================
# 2. 交互：时间范围（小时，相对各文件第一条记录）
# =============================
def prompt_time_range_h():
    """
    运行后由用户输入横轴时间范围；圈数多、总长不一时可每次设不同区间。
    返回 (t_start_h_or_None, t_end_h_or_None)：
      t_start_h_or_None 为 None 表示「每个文件从该文件 Time_h 的最小值开始」（通常约 0）。
      t_end_h_or_None 为 None 表示「每个文件用该文件自己的最大时间」。
    非 None 的起始时间可任意填写（如 48、120.5），不必从 0 开始。
    """
    print("\n—— 设置电压–时间图横轴范围（单位：小时，相对测试开始）——")
    print("  说明：起始可填任意时刻（例如从第 48 h 起看），不必从 0 开始；")
    print("        起始处直接回车 = 从该文件数据的最小时间画起（一般接近 0）。")
    s0 = input("  起始时间 (h)，回车 = 各文件数据最小时间：").strip()
    s1 = input("  结束时间 (h)，回车 = 各文件数据最大时间：").strip()

    if s0:
        try:
            t_start = float(s0)
        except ValueError:
            print("⚠️ 起始时间不是合法数字，已改为「按各文件最小时间」")
            t_start = None
    else:
        t_start = None

    if s1:
        try:
            t_end = float(s1)
        except ValueError:
            print("⚠️ 结束时间不是合法数字，已改为「按各文件最大时间」")
            t_end = None
    else:
        t_end = None

    if t_start is not None and t_end is not None and t_end < t_start:
        print("⚠️ 结束时间小于起始时间，已交换两者")
        t_start, t_end = t_end, t_start

    return t_start, t_end


def summarize_time_ranges(files):
    """
    读入每个 .ndax，计算相对起点的小时数，打印各文件 Time_h 的最小值与最大值。
    返回 [(file_name, df), ...]；df 已含列 Time_h，供后续直接作图（避免重复读盘）。
    """
    loaded = []
    printed_cols = False

    print("\n【数据时间范围（单位：h，相对各文件第一条记录）】")
    for file_name in sorted(files):
        try:
            df = NewareNDA.read(file_name)
            if 'Timestamp' not in df.columns or 'Voltage' not in df.columns:
                print(f"  ⚠️ 跳过 {file_name}：缺少 Timestamp 或 Voltage，列名：{list(df.columns)}")
                continue

            df = df.copy()
            t0 = df['Timestamp'].iloc[0]
            df['Time_h'] = (df['Timestamp'] - t0).dt.total_seconds() / 3600
            tmin = float(df['Time_h'].min())
            tmax = float(df['Time_h'].max())

            if not printed_cols:
                print("  列名（首个有效文件）：", list(df.columns))
                printed_cols = True

            print(f"  {file_name}:  最小 {tmin:.6g} h  →  最大 {tmax:.6g} h")

            loaded.append((file_name, df))

        except Exception as e:
            print(f"  ❌ 读取失败 {file_name}: {e}")

    if not loaded:
        return []

    if len(loaded) >= 2:
        mins_h = [float(d['Time_h'].min()) for _, d in loaded]
        maxs_h = [float(d['Time_h'].max()) for _, d in loaded]
        print(
            f"\n  汇总（所有文件）：各文件里「最小时间」的最小 = {min(mins_h):.6g} h，"
            f"各文件里「最大时间」的最大 = {max(maxs_h):.6g} h"
        )
        print("  （一般起点为 0，上排数字多为 0；选区间时对照每一行的最大时间，不要超过该文件的最大值。）")

    return loaded


def slice_df_by_time_h(df, t_start_h, t_end_h):
    """
    按 Time_h 截取子集。
    t_start_h 为 None 时从该文件 Time_h 最小值起；否则从 max(用户起始, 数据最小) 起。
    t_end_h 为 None 时用该表最大时间。
    返回 (df_plot, 实际横轴下限, 实际横轴上限)；与数据交集后仍无点则 df_plot 为空。
    """
    t_hi = float(df['Time_h'].max()) if t_end_h is None else min(float(t_end_h), float(df['Time_h'].max()))
    t_data_min = float(df['Time_h'].min())
    if t_start_h is None:
        t_lo = t_data_min
    else:
        t_lo = max(float(t_start_h), t_data_min)
    if t_lo > t_hi:
        return None, t_lo, t_hi
    m = (df['Time_h'] >= t_lo) & (df['Time_h'] <= t_hi)
    out = df.loc[m].copy()
    return out, t_lo, t_hi


# =============================
# 3. 核心函数
# =============================
def draw_real_voltage_time():

    path = os.getcwd()
    files = [f for f in os.listdir(path) if f.endswith('.ndax')]

    if not files:
        print("❌ 没有 .ndax 文件，别幻想了")
        return

    loaded = summarize_time_ranges(files)
    if not loaded:
        print("❌ 没有可处理的数据文件")
        return

    t_start_h, t_end_h = prompt_time_range_h()
    _parts = []
    if t_start_h is None:
        _parts.append("起始 = 各文件数据最小时间")
    else:
        _parts.append(f"起始 = {t_start_h:.6g} h 起（若早于数据起点会自动对齐到数据最小时间）")
    if t_end_h is None:
        _parts.append("结束 = 各文件自身最大时间")
    else:
        _parts.append(f"结束 = {t_end_h:.6g} h（若超过数据终点会截断到该文件最大时间）")
    print("已选：" + "；".join(_parts) + "\n")

    output_dir = os.path.join(path, "VT_time_choose_Results")
    os.makedirs(output_dir, exist_ok=True)

    for file_name, df in loaded:

        print(f"\n🔄 作图: {file_name}")

        try:
            vol_col = 'Voltage'

            # ======== 按用户输入的时间范围截取（df 已在扫描阶段带上 Time_h）========
            df_plot, t_xmin, t_xmax = slice_df_by_time_h(df, t_start_h, t_end_h)
            if df_plot is None or df_plot.empty:
                print(f"⚠️ 时间范围内无数据，跳过: {file_name}（请检查起始/结束时间）")
                continue

            # ======== 画图 ========
            fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

            ax.plot(df_plot['Time_h'], df_plot[vol_col],
                    color='#1f77b4',
                    linewidth=1.0)

            apply_style(ax)

            ax.set_xlabel('Time (h)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Voltage (V)', fontsize=14, fontweight='bold')

            ax.set_xlim(t_xmin, t_xmax)
            ax.set_ylim(2, 5)
       

            plt.tight_layout()

            save_path = os.path.join(
                output_dir,
                file_name.replace('.ndax', '_RealVT.png')
            )

            plt.savefig(save_path)
            plt.close()

            print(f"✅ 已生成: {save_path}")

        except Exception as e:
            print(f"❌ 报错: {e}")

# =============================
# 4. 运行
# =============================
if __name__ == "__main__":
    draw_real_voltage_time()