from scipy.special import factorial
from typing import Tuple
from cst_modeling.section import cst_foil  # 导入生成翼型坐标函数
import numpy as np
import matplotlib.pyplot as plt
from cst_modeling.math import cst_curve, interp_from_curve, find_circle_3p, clustcos
from cst_modeling.section import cst_foil_fit

def enhanced_cst_foil(nn: int, cst_base_u, cst_base_l, cst_le_u=None, cst_le_l=None, 
                    x=None, t=None, tail=0.0, xn1=0.5, xn2=1.0):
    '''
    使用CST方法构建具有增强前缘拟合能力的翼型
    
    基于图片所示方法，采用基础翼型与前缘修型叠加的方式构建翼型：
    1. 使用6阶伯恩斯坦多项式(n₀=7)构造基础翼型
    2. 使用39阶伯恩斯坦多项式(n₁=40)构造前缘修型的叠加部分
    3. 在叠加部分，仅前三个参数有效，其余参数为0
    
    参数:
    -----------
    nn: int
        生成点的总数量
    cst_base_u, cst_base_l: list or ndarray
        基础翼型的上下表面CST参数
    cst_le_u, cst_le_l: list or ndarray, optional
        前缘修型的上下表面CST参数(仅使用前三个参数)
    x: ndarray [nn], optional
        x坐标分布(范围[0,1])
    t: float, optional
        指定的相对最大厚度
    tail: float, optional
        尾缘相对厚度
    xn1, xn2: float
        CST形状参数
        
    返回:
    --------
    x, yu, yl: ndarray
        翼型坐标
    t0: float
        实际相对最大厚度
    R0: float
        前缘半径
    
    示例:
    ---------
    >>> x_, yu, yl, t0, R0 = enhanced_cst_foil(nn, cst_base_u, cst_base_l, cst_le_u, cst_le_l)
    '''
    from numpy import zeros, array, argmax, power

    # 确保参数为numpy数组
    cst_base_u = array(cst_base_u)
    cst_base_l = array(cst_base_l)
    
    # 生成基础坐标分布
    if x is None:
        x = zeros(nn)
        for i in range(nn):
            x[i] = clustcos(i, nn)
    
    # 生成基础翼型的上下表面曲线
    _, yu_base = cst_curve(nn, cst_base_u, x=x, xn1=xn1, xn2=xn2)
    _, yl_base = cst_curve(nn, cst_base_l, x=x, xn1=xn1, xn2=xn2)
    
    # 初始化最终曲线为基础曲线
    yu = yu_base.copy()
    yl = yl_base.copy()
    
    # 如果提供了前缘修型参数，计算并添加修型部分
    if cst_le_u is not None and len(cst_le_u) >= 3:
        # 创建前缘修型的CST参数数组(仅使用前三个参数)
        n_le = 40  # 39阶伯恩斯坦多项式
        cst_le_u_full = zeros(n_le)
        cst_le_u_full[:3] = array(cst_le_u[:3])
        
        # 计算前缘修型的上表面曲线
        _, yu_le = cst_curve(nn, cst_le_u_full, x=x, xn1=xn1, xn2=xn2)
        
        # 添加到基础曲线
        yu = yu_base + yu_le
    
    if cst_le_l is not None and len(cst_le_l) >= 3:
        # 创建前缘修型的CST参数数组(仅使用前三个参数)
        n_le = 40  # 39阶伯恩斯坦多项式
        cst_le_l_full = zeros(n_le)
        cst_le_l_full[:3] = array(cst_le_l[:3])
        
        # 计算前缘修型的下表面曲线
        _, yl_le = cst_curve(nn, cst_le_l_full, x=x, xn1=xn1, xn2=xn2)
        
        # 添加到基础曲线
        yl = yl_base + yl_le
    
    # 计算厚度
    thick = yu - yl
    it = argmax(thick)
    t0 = thick[it]
    
    # 应用厚度约束
    if t is not None:
        r = (t - tail * x[it]) / t0
        t0 = t
        yu = yu * r
        yl = yl * r
    
    # 添加尾缘厚度
    for i in range(nn):
        yu[i] += 0.5 * tail * x[i]
        yl[i] -= 0.5 * tail * x[i]
    
    # 在添加尾缘后更新t0
    if t is None:
        thick = yu - yl
        it = argmax(thick)
        t0 = thick[it]
    
    # 计算前缘半径
    x_RLE = 0.005
    yu_RLE = interp_from_curve(x_RLE, x, yu)
    yl_RLE = interp_from_curve(x_RLE, x, yl)
    R0, _ = find_circle_3p([0.0, 0.0], [x_RLE, yu_RLE], [x_RLE, yl_RLE])
    
    return x, yu, yl, t0, R0



def enhanced_le_cst_fit(xu: np.ndarray, yu: np.ndarray, xl: np.ndarray, yl: np.ndarray, 
                      n_base=7, n_diff=20, xn1=0.5, xn2=1.0):
    '''
    使用差异叠加的方法获得基础CST参数和前缘修型参数
    
    实现步骤：
    1. 首先使用n_base阶伯恩斯坦多项式拟合基础翼型参数
    2. 计算基础模型与原始翼型之间的差异
    3. 使用n_diff阶伯恩斯坦多项式拟合这个差异曲线
    4. 取差异模型的前三个参数作为前缘修型参数
    
    参数:
    -----------
    xu, yu: ndarray
        上表面x和y坐标
    xl, yl: ndarray
        下表面x和y坐标
    n_base: int
        基础翼型的CST参数阶数（默认为7阶伯恩斯坦多项式）
    n_diff: int
        差异曲线的CST参数阶数（默认为20阶伯恩斯坦多项式）
    xn1, xn2: float
        CST形状参数
        
    返回:
    --------
    cst_base_u, cst_base_l: ndarray
        基础翼型的上下表面CST参数
    cst_le_u, cst_le_l: ndarray
        前缘修型的上下表面CST参数（仅前三个）
    '''
    from scipy.special import factorial
    if xu[0] != 0 or yu[0] != 0:
        # 计算偏移量
        x_offset = xu[0]
        y_offset = yu[0]
        
        # 平移坐标
        xu = xu - x_offset
        yu = yu - y_offset
        xl = xl - x_offset
        yl = yl - y_offset
        
        print(f"前缘点已从({x_offset:.6f}, {y_offset:.6f})平移至(0, 0)")
    
    # 1. 获取基础翼型的CST参数（7阶伯恩斯坦多项式）
    cst_base_u, cst_base_l = cst_foil_fit(xu, yu, xl, yl, n_cst=n_base, xn1=xn1, xn2=xn2)
    
    # 2. 使用基础CST参数生成基础翼型
    nn1 = xu.shape[0]
    _, yu_base = cst_curve(nn1, cst_base_u, x=xu, xn1=xn1, xn2=xn2)
    nn2 = xl.shape[0]
    _, yl_base = cst_curve(nn2, cst_base_l, x=xl, xn1=xn1, xn2=xn2)
    
    # 3. 计算差异曲线
    yu_diff = yu - yu_base
    yl_diff = yl - yl_base
    
    # 4. 拟合差异曲线（用fit_curve而不是cst_foil_fit，因为要单独拟合每个表面）
    cst_diff_u = fit_curve_custom(xu, yu_diff, n_en=3, n_cst=n_diff, xn1=xn1, xn2=xn2)
    cst_diff_l = fit_curve_custom(xl, yl_diff, n_en=3, n_cst=n_diff, xn1=xn1, xn2=xn2)
    
    # 5. 仅取差异模型的前三个参数作为前缘修型参数
    cst_le_u = cst_diff_u[:3]
    cst_le_l = cst_diff_l[:3]
    
    return cst_base_u, cst_base_l, cst_le_u, cst_le_l

def fit_curve_custom(x: np.ndarray, y: np.ndarray,n_en =3, n_cst=20, xn1=0.5, xn2=1.0):
    '''
    使用最小二乘法拟合CST曲线
    '''
    # 确保x是从0到1的范围
    x_norm = (x - x.min()) / (x.max() - x.min())
    
    # 构建系数矩阵A和目标向量b
    nn = x.shape[0]
    A = np.zeros((nn, n_cst))
    b = y.copy()
    
    for ip in range(nn):
        C_n1n2 = np.power(x_norm[ip], xn1) * np.power(1-x_norm[ip], xn2)
        for i in range(n_cst):
            if i < n_en:    
                xk_i_n = factorial(n_cst-1)/factorial(i)/factorial(n_cst-1-i)
                A[ip, i] = xk_i_n * np.power(x_norm[ip], i) * np.power(1-x_norm[ip], n_cst-1-i) * C_n1n2
            else:
                A[ip, i] = 0
    
    # 使用最小二乘法求解
    solution = np.linalg.lstsq(A, b, rcond=None)
    return solution[0]

def enhanced_cst_foil_with_fit(xu: np.ndarray, yu: np.ndarray, xl: np.ndarray, yl: np.ndarray,
                             nn=101, t=None, tail=0.0, xn1=0.5, xn2=1.0):
    '''
    从翼型坐标自动拟合CST参数并生成具有增强前缘拟合的翼型
    
    参数:
    -----------
    xu, yu: ndarray
        上表面x和y坐标
    xl, yl: ndarray
        下表面x和y坐标
    nn: int
        生成点的总数量
    t: float, optional
        指定的相对最大厚度
    tail: float, optional
        尾缘相对厚度
    xn1, xn2: float
        CST形状参数
        
    返回:
    --------
    x, yu, yl: ndarray
        生成的翼型坐标
    t0: float
        实际相对最大厚度
    R0: float
        前缘半径
    cst_base_u, cst_base_l: ndarray
        基础翼型的上下表面CST参数
    cst_le_u, cst_le_l: ndarray
        前缘修型的上下表面CST参数（仅前三个）
    '''
    # 首先拟合获得基础和前缘修型的CST参数
    cst_base_u, cst_base_l, cst_le_u, cst_le_l = enhanced_le_cst_fit(
        xu, yu, xl, yl, n_base=7, n_diff=20, xn1=xn1, xn2=xn2)
    
    # 使用拟合的参数生成增强前缘的翼型
    x, yu_new, yl_new, t0, R0 = enhanced_cst_foil(
        nn, cst_base_u, cst_base_l, cst_le_u, cst_le_l, 
        x=None, t=t, tail=tail, xn1=xn1, xn2=xn2)
    
    return x, yu_new, yl_new, t0, R0, cst_base_u, cst_base_l, cst_le_u, cst_le_l

def read_coordinates(coord_path):
    """
    从指定的坐标文件中读取数据，并分离上下表面坐标。
    
    参数:
        coord_path: 坐标文件路径。
    
    返回:
        xu0, yu0, xl0, yl0: 上下表面的 x、y 坐标（均为 numpy 数组）
    """
    with open(coord_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    x_upper = []
    y_upper = []
    x_lower = []
    y_lower = []
    reading_upper = True  # 标记当前读取上表面

    for line in lines[3:]:  # 假定文件前3行为标题或无用信息
        if not line.strip():
            reading_upper = False
            continue

        x_, y_ = line.split()
        if reading_upper:
            x_upper.append(float(x_))
            y_upper.append(float(y_))
        else:
            x_lower.append(float(x_))
            y_lower.append(float(y_))

    xu0 = np.array(x_upper)
    yu0 = np.array(y_upper)
    xl0 = np.array(x_lower)
    yl0 = np.array(y_lower)

    # 如果起始点不为 0，则在开头插入 (0, 0)
    if xu0[0] != 0:
        xu0 = np.insert(xu0, 0, 0)
        yu0 = np.insert(yu0, 0, 0)
    if xl0[0] != 0:
        xl0 = np.insert(xl0, 0, 0)
        yl0 = np.insert(yl0, 0, 0)
        
    return xu0, yu0, xl0, yl0
def plot_cst_basis_functions(n_cst=10, nn=200, xn1=0.5, xn2=1.0):
    """
    可视化CST基函数的每个维度
    
    Parameters:
    -----------
    n_cst: int
        CST系数的维度数
    nn: int
        点的数量
    xn1, xn2: float
        CST参数
    """
    # 创建x坐标
    x = np.linspace(0, 1, nn)
    
    # 创建图形
    plt.figure(figsize=(12, 6))
    
    # 为每个维度计算并绘制基函数
    for i in range(n_cst):
        # 创建一个只在第i个位置为1的系数向量
        coef = np.zeros(n_cst)
        coef[i] = 1.0
        
        # 计算对应的曲线
        _, y = cst_curve(nn, coef, x, xn1, xn2)
        
        # 绘制曲线
        plt.plot(x, y, '-', label=f'Basis {i+1}')
    
    # 设置图形属性
    plt.title('CST Basis Functions')
    plt.xlabel(r'$\varphi$')
    plt.ylabel('Y($\varphi$)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(ncol=2)  # 将图例分成两列显示
    
    # 添加坐标轴
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    
    plt.show()
    
def visualize_enhanced_cst_fit(xu: np.ndarray, yu: np.ndarray, xl: np.ndarray, yl: np.ndarray, n_base=7, n_diff=20, n_enhanced=3):
    '''
    Visualize the enhanced CST fitting process
    '''
    # 1. 获取基础翼型的CST参数
    if xu[0] != 0 or yu[0] != 0:
        # 计算偏移量
        x_offset = xu[0]
        y_offset = yu[0]
        
        # 平移坐标
        xu = xu - x_offset
        yu = yu - y_offset
        xl = xl - x_offset
        yl = yl - y_offset
        
        print(f"前缘点已从({x_offset:.6f}, {y_offset:.6f})平移至(0, 0)")
    
    cst_base_u, cst_base_l = cst_foil_fit(xu, yu, xl, yl, n_cst=n_base)
    
    # 2. 使用基础CST参数生成基础翼型
    nn1 = xu.shape[0]
    _, yu_base = cst_curve(nn1, cst_base_u, x=xu)
    nn2 = xl.shape[0]
    _, yl_base = cst_curve(nn2, cst_base_l, x=xl)
    
    # 3. 计算差异曲线（仅前缘区域x<0.1）
    yu_diff = np.zeros_like(yu)
    yl_diff = np.zeros_like(yl)
    
    # 仅计算前缘区域(x<0.1)的差异
    #yu_diff[xu < 0.1] = yu[xu < 0.1] - yu_base[xu < 0.1]
    #yl_diff[xl < 0.1] = yl[xl < 0.1] - yl_base[xl < 0.1]
    yu_diff = yu - yu_base
    yl_diff = yl - yl_base
    
    # 4. 拟合差异曲线
    cst_diff_u = fit_curve_custom(xu, yu_diff, n_en=n_enhanced, n_cst=n_diff)
    cst_diff_l = fit_curve_custom(xl, yl_diff, n_en=n_enhanced, n_cst=n_diff)
    
    # 5. 生成拟合后的差异曲线
    _, yu_diff_fit = cst_curve(nn1, cst_diff_u, x=xu)
    _, yl_diff_fit = cst_curve(nn2, cst_diff_l, x=xl)
    
    # 6. 取差异模型的前n_enhanced个参数
    cst_le_u = cst_diff_u[:n_enhanced]
    cst_le_l = cst_diff_l[:n_enhanced]
    
    # 7. 使用前n_enhanced个参数生成高阶差异曲线
    n_le = n_diff
    cst_le_u_full = np.zeros(n_le)
    cst_le_u_full[:n_enhanced] = cst_le_u
    _, yu_le = cst_curve(nn1, cst_le_u_full, x=xu)
    
    cst_le_l_full = np.zeros(n_le)
    cst_le_l_full[:n_enhanced] = cst_le_l
    _, yl_le = cst_curve(nn2, cst_le_l_full, x=xl)
    
    # 8. 生成最终曲线
    yu_final =  yu_le + yu_base
    yl_final =  yl_le + yl_base
    
    # 创建图形
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # 原始翼型与基础拟合曲线对比
    axs[0, 0].plot(xu, yu, 'b-', label='Original Upper')
    axs[0, 0].plot(xl, yl, 'r-', label='Original Lower')
    axs[0, 0].plot(xu, yu_base, 'b--', label=f'Base Fit Upper (N={n_base})')
    axs[0, 0].plot(xl, yl_base, 'r--', label=f'Base Fit Lower (N={n_base})')
    axs[0, 0].set_title(f'Original Airfoil vs Base Fit (N={n_base})')
    axs[0, 0].legend()
    axs[0, 0].set_aspect('equal')
    
    # 差异曲线
    axs[0, 1].plot(xu, yu_diff, 'b-', label='Upper Difference')
    axs[0, 1].plot(xl, yl_diff, 'r-', label='Lower Difference')
    axs[0, 1].plot(xu, yu_diff_fit, 'b--', label=f'Diff Fit Upper (N={n_diff})')
    axs[0, 1].plot(xl, yl_diff_fit, 'r--', label=f'Diff Fit Lower (N={n_diff})')
    axs[0, 1].set_title(f'Difference Curves and Fits (N={n_diff})')
    axs[0, 1].legend()
    
    # 仅前三个参数的高阶差异曲线
    axs[1, 0].plot(xu, yu_diff, 'b-', label='Upper Difference')
    axs[1, 0].plot(xl, yl_diff, 'r-', label='Lower Difference')
    axs[1, 0].plot(xu, yu_le, 'b--', label='First 3 Params Upper')
    axs[1, 0].plot(xl, yl_le, 'r--', label='First 3 Params Lower')
    axs[1, 0].set_title('Difference vs First 3 Parameters Fit')
    axs[1, 0].legend()
    
    # 最终拟合翼型
    axs[1, 1].plot(xu, yu, 'b-', label='Original Upper')
    axs[1, 1].plot(xl, yl, 'r-', label='Original Lower')
    axs[1, 1].plot(xu, yu_final, 'b--', label='Enhanced Fit Upper')
    axs[1, 1].plot(xl, yl_final, 'r--', label='Enhanced Fit Lower')
    axs[1, 1].set_title('Original vs Enhanced CST Fit')
    axs[1, 1].legend()
    axs[1, 1].set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    
    # 放大显示前缘区域
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=120)

    # 定义更美观的颜色方案
    colors = {
        'original_upper': '#0066CC',  # Deep blue
        'original_lower': '#CC3300',  # Deep red
        'base_upper': '#66CCFF',      # Light blue
        'base_lower': '#FF9966',      # Light red/orange
        'enhanced_upper': '#3366FF',  # Blue-purple
        'enhanced_lower': '#FF6666'   # Pink
    }

    # 绘制第一幅图 - 基础版本
    le_range = 0.1  # 前缘范围
    axs[0].plot(xu[xu <= le_range], yu[xu <= le_range], '-', color=colors['original_upper'], 
               linewidth=3, label='Original Upper')
    axs[0].plot(xl[xl <= le_range], yl[xl <= le_range], '-', color=colors['original_lower'], 
               linewidth=3, label='Original Lower')
    axs[0].plot(xu[xu <= le_range], yu_base[xu <= le_range], '--', color=colors['base_upper'], 
               linewidth=2.5, label='Base Fit Upper')
    axs[0].plot(xl[xl <= le_range], yl_base[xl <= le_range], '--', color=colors['base_lower'], 
               linewidth=2.5, label='Base Fit Lower')

    # 设置标题和坐标轴
    axs[0].set_title('Leading Edge: Original vs Base Fit', fontsize=14, fontweight='bold')
    axs[0].set_xlabel('x/c', fontsize=12)
    axs[0].set_ylabel('y/c', fontsize=12)
    axs[0].grid(True, linestyle='--', alpha=0.5)
    axs[0].legend(loc='best', fontsize=10, framealpha=0.7)
    axs[0].set_aspect('equal')

    # 添加坐标轴
    axs[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    axs[0].axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

    # 绘制第二幅图 - 原来的enhanced fit版本
    axs[1].plot(xu[xu <= le_range], yu[xu <= le_range], '-', color=colors['original_upper'], 
               linewidth=3, label='Original Upper')
    axs[1].plot(xl[xl <= le_range], yl[xl <= le_range], '-', color=colors['original_lower'], 
               linewidth=3, label='Original Lower')
    axs[1].plot(xu[xu <= le_range], yu_final[xu <= le_range], '--', color=colors['enhanced_upper'], 
               linewidth=2.5, label='Enhanced Fit Upper')
    axs[1].plot(xl[xl <= le_range], yl_final[xl <= le_range], '--', color=colors['enhanced_lower'], 
               linewidth=2.5, label='Enhanced Fit Lower')

    # 设置标题和坐标轴
    axs[1].set_title('Leading Edge: Original vs Enhanced Fit', fontsize=14, fontweight='bold')
    axs[1].set_xlabel('x/c', fontsize=12)
    axs[1].set_ylabel('y/c', fontsize=12)
    axs[1].grid(True, linestyle='--', alpha=0.5)
    axs[1].legend(loc='best', fontsize=10, framealpha=0.7)
    #axs[1].set_aspect('equal')

    # 添加坐标轴
    axs[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    axs[1].axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

    # 为高精度对比图创建一个新的图形
    plt.figure(figsize=(10, 8), dpi=150)

    # 前缘区域范围
    le_range = 0.015

    # 生成高精度曲线点
    nn_smooth = 4001
    x_smooth, yu_base_smooth = cst_curve(nn_smooth, cst_base_u)
    x_smooth, yl_base_smooth = cst_curve(nn_smooth, cst_base_l)

    # 生成增强版前缘修型
    n_le = n_diff
    cst_le_u_full = np.zeros(n_le)
    cst_le_u_full[:n_enhanced] = cst_le_u
    x_smooth, yu_le_smooth = cst_curve(nn_smooth, cst_le_u_full)

    cst_le_l_full = np.zeros(n_le)
    cst_le_l_full[:n_enhanced] = cst_le_l
    x_smooth, yl_le_smooth = cst_curve(nn_smooth, cst_le_l_full)

    # 计算最终曲线
    yu_final_smooth = yu_base_smooth + yu_le_smooth
    yl_final_smooth = yl_base_smooth + yl_le_smooth

    # 更美观的颜色方案
    colors = {
        'original_points': '#777777',     # Medium gray for original points
        'base_fit': '#005A87',            # Deep navy blue for standard CST
        'enhanced_fit': '#B83A34'         # Soft burgundy red for enhanced CST
    }

    # 绘制原始数据点
    plt.scatter(xu[xu <= le_range], yu[xu <= le_range], s=25, color=colors['original_points'], 
          marker='s', alpha=0.9, label='Original Points (Whitcomb)')
    plt.scatter(xl[xl <= le_range], yl[xl <= le_range], s=25, color=colors['original_points'], 
          marker='s', alpha=0.9, label='')

    # 绘制平滑曲线 - 基础拟合用蓝色实线
    plt.plot(x_smooth[x_smooth <= le_range], yu_base_smooth[x_smooth <= le_range], '-', 
           color=colors['base_fit'], linewidth=1.8, label='Base CST Fit')
    plt.plot(x_smooth[x_smooth <= le_range], yl_base_smooth[x_smooth <= le_range], '-', 
           color=colors['base_fit'], linewidth=1.8)

    # 绘制平滑曲线 - 增强拟合用红色虚线
    plt.plot(x_smooth[x_smooth <= le_range], yu_final_smooth[x_smooth <= le_range], '--', 
           color=colors['enhanced_fit'], linewidth=1.8, label='Enhanced CST Fit')
    plt.plot(x_smooth[x_smooth <= le_range], yl_final_smooth[x_smooth <= le_range], '--', 
           color=colors['enhanced_fit'], linewidth=1.8)

    # 设置标题和坐标轴
    plt.title('Leading Edge: Base CST vs Enhanced CST', fontsize=14, fontweight='bold')
    plt.xlabel('$x/c$', fontsize=15)
    plt.ylabel('$y/c$', fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='best', fontsize=13, framealpha=0.7)
    #plt.gca().set_aspect('equal')

    # 添加坐标轴
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

    # 调整布局
    plt.tight_layout()

    # 添加注释说明
    plt.savefig(f'./img/{file}_leading_edge_comparison.png', dpi=300)
    plt.show()
    return cst_base_u, cst_base_l, cst_le_u, cst_le_l

# 使用读取的翼型坐标进行可视化
file = "./whitcomb-il.dat"
xu0, yu0, xl0, yl0 = read_coordinates(file)
n_diff = 40
cst_base_u, cst_base_l, cst_le_u, cst_le_l = visualize_enhanced_cst_fit(xu0, yu0, xl0, yl0,n_base=10,n_diff=n_diff,n_enhanced=3)
print(cst_base_u)
print(cst_base_l)
print(cst_le_u)
print(cst_le_l)
def plot_cst_basis_comparison(cst_base, cst_le):
    """
    对比7阶和20阶CST基函数
    """
    # 创建x坐标
    nn = 200
    x = np.linspace(0, 1, nn)
    
    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # 绘制7阶CST基函数
    n_cst_7 = 7
    colors_7 = plt.cm.tab10(np.linspace(0, 1, n_cst_7))
    
    for i in range(n_cst_7):
        coef = np.zeros(n_cst_7)
        coef[i] = cst_base[i]
        _, y = cst_curve(nn, coef, x)
        ax1.plot(x, y, '-', color=colors_7[i], label=f'i={i+1}')
    
    ax1.set_title('7th Order CST Basis Functions', fontsize=12)
    ax1.set_xlabel('φ', fontsize=10)
    ax1.set_ylabel('Y(φ)', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(ncol=4, fontsize=9)
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax1.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    
    # 绘制20阶CST基函数
    n_cst_20 = len(cst_le)
    colors_20 = plt.cm.rainbow(np.linspace(0, 1, n_cst_20))
    
    for i in range(n_cst_20):
        coef = np.zeros(n_cst_20)
        coef[i] = cst_le[i]
        _, y = cst_curve(nn, coef, x)
        ax2.plot(x, y, '-', color=colors_20[i], label=f'i={i+1}')
    
    ax2.set_title('20th Order CST Basis Functions', fontsize=12)
    #ax2.set_xlim(0, 0.1)
    #ax2.set_ylim(-0.05, 0.05)
    ax2.set_xlabel('φ', fontsize=10)
    ax2.set_ylabel('Y(φ)', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(ncol=5, fontsize=8)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()
    
    # 绘制前三个基函数的对比
    plt.figure(figsize=(12, 6))
    
    # 7阶的前三个基函数
    for i in range(n_cst_7):
        coef = np.zeros(n_cst_7)
        coef[i] = cst_base[i]
        _, y = cst_curve(nn, coef, x)
        plt.plot(x, y, '--', color=colors_7[i], label=f'N=7, i={i+1}')
    
    # 20阶的前三个基函数
    for i in range(3):
        coef = np.zeros(n_cst_20)
        coef[i] = cst_le[i]
        _, y = cst_curve(nn, coef, x)
        plt.plot(x, y, '-', color=colors_20[i], label=f'N=20, i={i+1}')
    
    plt.title('Comparison of First 3 Basis Functions (7th vs 20th Order)', fontsize=12)
    plt.ylim(-0.05, 0.05)
    plt.xlabel('φ', fontsize=10)
    plt.ylabel('Y(φ)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()

    plt.show()

# 调用函数绘制图形
#plot_cst_basis_comparison(cst_base_u, cst_le_u)
