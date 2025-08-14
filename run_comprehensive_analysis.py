import numpy as np
import matplotlib.pyplot as plt
import os

# --- Thiết lập Môi trường ---
output_dir = "figures_v2"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- Bước 1: Định nghĩa các hàm chi phí (Cost Functions) ---

def cost_dijkstra(n, m):
    """Chi phí của Dijkstra với Fibonacci Heap: O(m + n*log(n))"""
    return m + n * np.log2(n)

def cost_duan_et_al(n, m):
    """Chi phí của thuật toán cổ điển mới: O(m * (log n)^(2/3))"""
    with np.errstate(divide='ignore', invalid='ignore'):
        log_val = np.log2(n)
        log_val[np.isneginf(log_val)] = 0
        return m * (log_val)**(2/3)

def cost_quantum_grover(n, m):
    """Chi phí của thuật toán SSSP lượng tử dựa trên Grover: O(sqrt(n) * m)"""
    return np.sqrt(n) * m

def cost_quantum_wesolowski(n, m, l):
    """Chi phí của thuật toán lượng tử Divide & Conquer: Õ(l * sqrt(m))"""
    # Õ bỏ qua các yếu tố log, nên chúng ta sẽ bỏ qua chúng trong mô hình
    return l * np.sqrt(m)

# --- Hàm trợ giúp để chạy và hiển thị kết quả ---

def run_and_plot_scenario(scenario_name, filename, n_values, m_func, l_func=None):
    """Hàm tổng quát để chạy một kịch bản, in kết quả và vẽ đồ thị."""
    
    print(f"\n--- SCENARIO: {scenario_name} ---\n")
    
    m_values = m_func(n_values)
    l_values = l_func(n_values) if l_func else None
    
    # Tính toán chi phí
    costs = {
        'Dijkstra': cost_dijkstra(n_values, m_values),
        'Duan et al.': cost_duan_et_al(n_values, m_values),
        'Grover SSSP': cost_quantum_grover(n_values, m_values)
    }
    if l_values is not None:
        costs['Wesolowski et al.'] = cost_quantum_wesolowski(n_values, m_values, l_values)

    # In kết quả tại các điểm tiêu biểu
    print(f"Theoretical Costs ({scenario_name}):")
    header = f"{'n':>10s}" + "".join([f" | {name:>20s}" for name in costs.keys()])
    print(header)
    print("-" * len(header))
    
    for n_point in [1e4, 1e6, 1e8]:
        if n_point > n_values.max() or n_point < n_values.min(): continue
        idx = np.argmin(np.abs(n_values - n_point))
        row = f"{n_values[idx]:>10.0e}"
        for name in costs.keys():
            row += f" | {costs[name][idx]:>20.2e}"
        print(row)
    print("\n")

    # Vẽ và lưu đồ thị
    plt.figure(figsize=(12, 8))
    
    plt.plot(n_values, costs['Dijkstra'], label='Dijkstra (Classical)', color='royalblue', linewidth=2)
    plt.plot(n_values, costs['Duan et al.'], label='Duan et al. (New Classical)', color='darkorange', linewidth=3.5)
    plt.plot(n_values, costs['Grover SSSP'], label='Grover SSSP (Quantum)', color='seagreen', linestyle='--', linewidth=2)
    if 'Wesolowski et al.' in costs:
        plt.plot(n_values, costs['Wesolowski et al.'], label='Wesolowski et al. (Quantum)', color='crimson', linestyle=':', linewidth=3.5)

    plt.xscale('log')
    plt.yscale('log')
    plt.title(f'Cost Comparison for SSSP ({scenario_name})', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Vertices (n)', fontsize=12)
    plt.ylabel('Theoretical Computational Cost (log scale)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, which="both", ls="--", color='gray', alpha=0.6)
    figure_path = os.path.join(output_dir, filename)
    plt.savefig(figure_path, dpi=600, bbox_inches='tight')
    print(f"Figure saved to: {figure_path}\n")
    plt.close()


# --- Bước 3: Định nghĩa và Chạy các Kịch bản ---

print("="*60)
print("COMPREHENSIVE ANALYSIS START")
print("="*60)

# Dải giá trị n chung
n_vals = np.logspace(3, 9, 100)

# Kịch bản 1: Đồ thị Thưa, Đường đi Ngắn
run_and_plot_scenario(
    scenario_name="Sparse Graph, Short Path",
    filename="sparse_short_path.png",
    n_values=n_vals,
    m_func=lambda n: 10 * n,
    l_func=lambda n: np.log2(n)**2 # Giả sử l tăng rất chậm, ví dụ (log n)^2
)

# Kịch bản 2: Đồ thị Thưa, Đường đi Dài
run_and_plot_scenario(
    scenario_name="Sparse Graph, Long Path",
    filename="sparse_long_path.png",
    n_values=n_vals,
    m_func=lambda n: 10 * n,
    l_func=lambda n: n / 10 # Giả sử l = n/10
)

# Dải giá trị n nhỏ hơn cho đồ thị dày
n_vals_dense = np.logspace(3, 5, 100)

# Kịch bản 3: Đồ thị Dày đặc, Đường đi Ngắn
run_and_plot_scenario(
    scenario_name="Dense Graph, Short Path",
    filename="dense_short_path.png",
    n_values=n_vals_dense,
    m_func=lambda n: n**2 / 100,
    l_func=lambda n: np.log2(n)**2
)

# Kịch bản 4: Đồ thị Dày đặc, Đường đi Dài
run_and_plot_scenario(
    scenario_name="Dense Graph, Long Path",
    filename="dense_long_path.png",
    n_values=n_vals_dense,
    m_func=lambda n: n**2 / 100,
    l_func=lambda n: n / 10
)

print("="*60)
print("COMPREHENSIVE ANALYSIS COMPLETE")
print("="*60)
