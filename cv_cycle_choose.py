import NewareNDA
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker

# 电极面积 (cm^2)，与容量归一化一致
AREA_CM2 = 1.13

# ==========================================
# 1. 样式（Arial；坐标轴标签不加粗）
# ==========================================
SPINE_LW = 0.8   # 坐标轴边框线宽
TICK_LW = 0.8    # 刻度线宽

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.linewidth': SPINE_LW,

    'axes.labelweight': 'normal',
    # 仅 mathtext 片段用 Arial，与正文一致；避免 Unicode 上标负号 U+207B 在 Arial 下方块
    'mathtext.fontset': 'custom',
    'mathtext.default': 'regular',
    'mathtext.rm': 'Arial',
    'mathtext.it': 'Arial:italic',
    'mathtext.bf': 'Arial:bold',
})

# 整句一条 mathtext，与「正文 + $...$ 片段」相比基线一致，上标 −2 不会整块被抬高；\cdot 为点乘；仍用 \mathrm{cm}^{-2} 避免 c·m^{-2}
# 若仍嫌上标略高，可把下面改成 ...\mathrm{cm}^{\!-2})}$（\! 为负细间距，略拉近上标）
XLABEL_AREAL_CAP = r'$\mathrm{Capacity\ (mAh\cdot\mathrm{cm}^{-2})}$'
YLABEL_VOLTAGE = 'Voltage (V)'


def apply_cap_style(ax):
    """四边封口，刻度向内，细线框"""
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(SPINE_LW)
    ax.tick_params(
        axis='x', direction='in', bottom=True, labelbottom=True,
        length=5, width=TICK_LW, labelsize=12,
    )
    ax.tick_params(
        axis='y', direction='in', left=True, labelleft=True,
        length=5, width=TICK_LW, labelsize=12,
    )


# ==========================================
# 2. 数据列与圈数
# ==========================================
def prepare_df_cv(df):
    """
    为 df 增加 Norm_Cap，并返回 (cycle_col, step_col, vol_col)；无法识别时返回 None。
    """
    cap_cols = [c for c in df.columns if 'Cap' in c]
    if not cap_cols:
        return None

    chg_col = next((c for c in cap_cols if 'Chg' in c or 'Charge' in c), None)
    dchg_col = next((c for c in cap_cols if 'DChg' in c or 'Discharge' in c), None)

    if chg_col and dchg_col:
        df['Norm_Cap'] = (df[chg_col].fillna(0) + df[dchg_col].fillna(0)) / AREA_CM2
    else:
        df['Norm_Cap'] = df[cap_cols[0]] / AREA_CM2

    cycle_col = next((c for c in df.columns if 'Cycle' in c), None)
    step_col = next((c for c in df.columns if 'Step' in c), None)
    vol_col = next((c for c in df.columns if 'Volt' in c), None)

    if not cycle_col or not step_col or not vol_col:
        return None
    return cycle_col, step_col, vol_col


def summarize_cycle_ranges(files):
    """
    读入每个 .ndax，打印各文件 Cycle 的最小圈与最大圈。
    返回 [(file_name, df, cycle_col, step_col, vol_col), ...]。
    """
    loaded = []
    printed_cols = False

    print("\n【数据圈数范围】")
    for file_name in sorted(files):
        try:
            df = NewareNDA.read(file_name)
            cols = prepare_df_cv(df)
            if cols is None:
                print(f"  ⚠️ 跳过 {file_name}：缺少容量/Cycle/Step/Voltage 等列")
                continue

            cycle_col, step_col, vol_col = cols
            cmin = int(df[cycle_col].min())
            cmax = int(df[cycle_col].max())

            if not printed_cols:
                print("  列名（首个有效文件）：", list(df.columns))
                printed_cols = True

            print(f"  {file_name}:  最小圈 {cmin}  →  最大圈 {cmax}")

            loaded.append((file_name, df, cycle_col, step_col, vol_col))

        except Exception as e:
            print(f"  ❌ 读取失败 {file_name}: {e}")

    if not loaded:
        return []

    if len(loaded) >= 2:
        mins_c = [int(d[c].min()) for _, d, c, _, _ in loaded]
        maxs_c = [int(d[c].max()) for _, d, c, _, _ in loaded]
        print(
            f"\n  汇总（所有文件）：最小圈数下限 = {min(mins_c)}，最大圈数上限 = {max(maxs_c)}"
        )
        print("  （选圈时请对照上表；结束圈不要超过该文件的最大圈。）")

    return loaded


def prompt_cycle_range():
    """
    返回 (c_start_or_None, c_end_or_None, step_int)。
    """
    print("\n—— 设置要画的循环圈数 ——")
    print("  说明：可填任意起止圈（如从第 10 圈画到第 50 圈）；")
    print("        起始处回车 = 从该文件的最小圈画起；结束处回车 = 画到该文件最大圈。")
    s0 = input("  起始圈数，回车 = 各文件最小圈：").strip()
    s1 = input("  结束圈数，回车 = 各文件最大圈：").strip()

    if s0:
        try:
            c_start = int(float(s0))
        except ValueError:
            print("⚠️ 起始圈不是合法整数，已改为「按各文件最小圈」")
            c_start = None
    else:
        c_start = None

    if s1:
        try:
            c_end = int(float(s1))
        except ValueError:
            print("⚠️ 结束圈不是合法整数，已改为「按各文件最大圈」")
            c_end = None
    else:
        c_end = None

    if c_start is not None and c_end is not None and c_end < c_start:
        print("⚠️ 结束圈小于起始圈，已交换两者")
        c_start, c_end = c_end, c_start

    print(
        "  说明：间隔 N 时必画范围内「第一圈」；再从「第一圈+N−1」起每隔 N 取一圈（若存在、且不超过尾圈），"
        "不要求一定画最后一圈。例：1～10、间隔 5 → 1,5,10；1～11、间隔 5 → 1,5,10；间隔 2 → 1,2,4,6,8,10。"
    )
    s2 = input("  间隔圈数，回车 = 1（起止范围内连续全画）：").strip()
    if s2:
        try:
            step = int(float(s2))
            if step < 1:
                print("⚠️ 间隔须 ≥ 1，已改为 1")
                step = 1
        except ValueError:
            print("⚠️ 间隔不是合法整数，已改为 1")
            step = 1
    else:
        step = 1

    return c_start, c_end, step


def cycle_indices_in_range(df, cycle_col, c_start, c_end):
    cmin = int(df[cycle_col].min())
    cmax = int(df[cycle_col].max())
    lo = cmin if c_start is None else max(c_start, cmin)
    hi = cmax if c_end is None else min(c_end, cmax)
    if lo > hi:
        return [], lo, hi

    seen = sorted({int(float(x)) for x in df[cycle_col].unique()})
    cycles = [c for c in seen if lo <= c <= hi]
    return cycles, lo, hi


def cycles_with_interval(cycles_in_range, lo, hi, step):
    """
    在 cycles_in_range（已排序、且落在 [lo,hi] 内）中按间隔取样：
      - 必含「第一圈」；再从「首圈 + (step−1)」起每隔 step 取一圈（若存在），直到超过范围内尾圈；
      - 不要求必含最后一圈（例：1～11、间隔 5 → 1,5,10）。
    例：1～10、step=5 → 1,5,10；step=2 → 1,2,4,6,8,10。
    """
    if step is None or int(step) < 1:
        step = 1
    step = int(step)
    if not cycles_in_range:
        return []
    if step == 1:
        return list(cycles_in_range)

    exist = set(cycles_in_range)
    first = cycles_in_range[0]
    last = cycles_in_range[-1]

    out_set = {first}
    s = first + step - 1
    while s <= last:
        if s in exist:
            out_set.add(s)
        s += step

    return [c for c in cycles_in_range if c in out_set]


# ==========================================
# 3. 核心绘图逻辑
# ==========================================
def draw_clean_normalized_plots():
    path = os.getcwd()
    files = [f for f in os.listdir(path) if f.endswith('.ndax')]

    if not files:
        print("❌ 未发现 .ndax 文件")
        return

    loaded = summarize_cycle_ranges(files)
    if not loaded:
        print("❌ 没有可处理的数据文件")
        return

    c_start, c_end, cycle_step = prompt_cycle_range()
    _parts = []
    if c_start is None:
        _parts.append("起始 = 各文件最小圈")
    else:
        _parts.append(f"起始 = 第 {c_start} 圈起（若小于数据最小圈会对齐到数据最小圈）")
    if c_end is None:
        _parts.append("结束 = 各文件最大圈")
    else:
        _parts.append(f"结束 = 第 {c_end} 圈（若超过数据最大圈会截断到该文件最大圈）")
    if cycle_step <= 1:
        _parts.append("间隔 = 每圈都画")
    else:
        _parts.append(
            f"间隔 = {cycle_step}（必含首圈；从首圈+{cycle_step - 1} 起每 {cycle_step} 圈一条，末圈不强制）"
        )
    print("已选：" + "；".join(_parts) + "\n")

    output_dir = os.path.join(path, "CV_cycle_gap_Plots")
    os.makedirs(output_dir, exist_ok=True)

    deep_colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e']

    for file_name, df, cycle_col, step_col, vol_col in loaded:
        print(f"\n🔄 作图: {file_name}")

        try:
            cycles_all, lo_eff, hi_eff = cycle_indices_in_range(df, cycle_col, c_start, c_end)
            cycles_to_draw = cycles_with_interval(cycles_all, lo_eff, hi_eff, cycle_step)
            if not cycles_to_draw:
                print(f"⚠️ 所选圈数/间隔下无数据，跳过: {file_name}")
                continue

            fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

            for idx, cycle_num in enumerate(cycles_to_draw):
                cycle_data = df[df[cycle_col] == cycle_num]
                if cycle_data.empty:
                    cycle_data = df[df[cycle_col].astype(float) == float(cycle_num)]
                current_color = deep_colors[idx % len(deep_colors)]
                label_set = False

                for s_id in cycle_data[step_col].unique():
                    s_data = cycle_data[cycle_data[step_col] == s_id]
                    if s_data['Norm_Cap'].max() - s_data['Norm_Cap'].min() < 1e-9:
                        continue

                    label = f'Cycle {cycle_num}' if not label_set else None
                    ax.plot(s_data['Norm_Cap'], s_data[vol_col],
                            color=current_color, linewidth=2.2, label=label)
                    label_set = True

            apply_cap_style(ax)
            ax.set_xlabel(XLABEL_AREAL_CAP, fontsize=14)
            ax.set_ylabel(YLABEL_VOLTAGE, fontsize=14, fontfamily='Arial')
            ax.set_ylim(1.5, 4.8)

            formatter = ticker.ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-3, 3))
            ax.xaxis.set_major_formatter(formatter)

            ax.legend(loc='best', fontsize=11, frameon=False)

            save_name = file_name.replace('.ndax', '_Areal_Clean.png')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, save_name))
            plt.close()
            print(
                f"✅ 已生成: {save_name}（范围内 {len(cycles_all)} 圈，"
                f"间隔 {cycle_step} 实际绘 {len(cycles_to_draw)} 圈）"
            )

        except Exception as e:
            print(f"❌ 处理 {file_name} 出错: {e}")


if __name__ == "__main__":
    draw_clean_normalized_plots()
