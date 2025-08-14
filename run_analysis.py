import numpy as np
import matplotlib.pyplot as plt
import os

# --- Thiết lập Môi trường ---
# Tạo một thư mục để lưu hình ảnh nếu nó chưa tồn tại
output_dir = "figures"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- Bước 1: Định nghĩa các hàm chi phí (Cost Functions) ---
# (Không thay đổi so với trước)
def cost_dijkstra(n, m):
    """Chi phí của Dijkstra với Fibonacci Heap: O(m + n*log(n))"""
    return m + n * np.log2(n)

def cost_duan_et_al(n, m):
    """Chi phí của thuật toán cổ điển mới: O(m * (log n)^(2/3))"""
    # Xử lý trường hợp n=1 để tránh lỗi log(1)=0
    with np.errstate(divide='ignore', invalid='ignore'):
        log_val = np.log2(n)
        log_val[np.isneginf(log_val)] = 0
        return m * (log_val)**(2/3)

def cost_quantum_grover(n, m):
    """Chi phí của thuật toán SSSP lượng tử dựa trên Grover: O(sqrt(n) * m)"""
    return np.sqrt(n) * m

# --- Bước 2: Chuẩn bị dữ liệu ---
n_values = np.logspace(3, 9, num=100) # Dải n rộng hơn: từ 10^3 (nghìn) đến 10^9 (tỷ)

# --- Bước 3: Thực nghiệm và Phân tích ---

print("="*50)
print("ANALYSIS START: Comparing SSSP Algorithm Costs")
print("="*50)

# === Kịch bản 1: Đồ thị Thưa (Sparse Graph, m = 10n) ===
print("\n--- SCENARIO 1: SPARSE GRAPHS (m = 10n) ---\n")
m_sparse = 10 * n_values

# Tính toán chi phí
costs_sparse = {
    'Dijkstra': cost_dijkstra(n_values, m_sparse),
    'Duan et al.': cost_duan_et_al(n_values, m_sparse),
    'Grover SSSP': cost_quantum_grover(n_values, m_sparse)
}

# In kết quả tại các điểm tiêu biểu
print("Theoretical Costs (Sparse):")
print(f"{'n':>10s} | {'Dijkstra':>20s} | {'Duan et al.':>20s} | {'Grover SSSP':>20s}")
print("-"*75)
for n_point in [1e3, 1e6, 1e9]:
    idx = np.argmin(np.abs(n_values - n_point))
    print(f"{n_values[idx]:>10.0e} | {costs_sparse['Dijkstra'][idx]:>20.2e} | {costs_sparse['Duan et al.'][idx]:>20.2e} | {costs_sparse['Grover SSSP'][idx]:>20.2e}")
print("\n")


# Vẽ và lưu đồ thị
plt.figure(figsize=(12, 8))
plt.plot(n_values, costs_sparse['Dijkstra'], label='Dijkstra (Classical Baseline)', color='royalblue', linewidth=2)
plt.plot(n_values, costs_sparse['Duan et al.'], label='Duan et al. (New Classical)', color='darkorange', linewidth=3.5)
plt.plot(n_values, costs_sparse['Grover SSSP'], label='Grover SSSP (Quantum)', color='seagreen', linestyle='--', linewidth=2)

plt.xscale('log')
plt.yscale('log')
plt.title('Cost Comparison for SSSP (Sparse Graphs, m=10n)', fontsize=16, fontweight='bold')
plt.xlabel('Number of Vertices (n)', fontsize=12)
plt.ylabel('Theoretical Computational Cost (log scale)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, which="both", ls="--", color='gray', alpha=0.6)
figure_path_sparse = os.path.join(output_dir, "sssp_comparison_sparse.png")
plt.savefig(figure_path_sparse, dpi=600, bbox_inches='tight')
print(f"Figure saved to: {figure_path_sparse}\n")
plt.close() # Đóng hình để giải phóng bộ nhớ


# === Kịch bản 2: Đồ thị Dày đặc (Dense Graph, m = n^2/100) ===
print("\n--- SCENARIO 2: DENSE GRAPHS (m = n²/100) ---\n")
n_values_dense = np.logspace(3, 5, num=100) # Dải n nhỏ hơn cho đồ thị dày: 10^3 -> 10^5
m_dense = (n_values_dense**2) / 100

# Tính toán chi phí
costs_dense = {
    'Dijkstra': cost_dijkstra(n_values_dense, m_dense),
    'Duan et al.': cost_duan_et_al(n_values_dense, m_dense),
    'Grover SSSP': cost_quantum_grover(n_values_dense, m_dense)
}

# In kết quả tại các điểm tiêu biểu
print("Theoretical Costs (Dense):")
print(f"{'n':>10s} | {'Dijkstra':>20s} | {'Duan et al.':>20s} | {'Grover SSSP':>20s}")
print("-"*75)
for n_point in [1e3, 1e4, 1e5]:
    idx = np.argmin(np.abs(n_values_dense - n_point))
    print(f"{n_values_dense[idx]:>10.0e} | {costs_dense['Dijkstra'][idx]:>20.2e} | {costs_dense['Duan et al.'][idx]:>20.2e} | {costs_dense['Grover SSSP'][idx]:>20.2e}")
print("\n")

# Vẽ và lưu đồ thị
plt.figure(figsize=(12, 8))
plt.plot(n_values_dense, costs_dense['Dijkstra'], label='Dijkstra (Classical Baseline)', color='royalblue', linewidth=2)
plt.plot(n_values_dense, costs_dense['Duan et al.'], label='Duan et al. (New Classical)', color='darkorange', linewidth=3.5)
plt.plot(n_values_dense, costs_dense['Grover SSSP'], label='Grover SSSP (Quantum)', color='seagreen', linestyle='--', linewidth=2)

plt.xscale('log')
plt.yscale('log')
plt.title('Cost Comparison for SSSP (Dense Graphs, m=n²/100)', fontsize=16, fontweight='bold')
plt.xlabel('Number of Vertices (n)', fontsize=12)
plt.ylabel('Theoretical Computational Cost (log scale)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, which="both", ls="--", color='gray', alpha=0.6)
figure_path_dense = os.path.join(output_dir, "sssp_comparison_dense.png")
plt.savefig(figure_path_dense, dpi=600, bbox_inches='tight')
print(f"Figure saved to: {figure_path_dense}\n")
plt.close()

print("="*50)
print("ANALYSIS COMPLETE")
print("="*50)
