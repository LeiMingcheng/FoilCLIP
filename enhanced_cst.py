from matplotlib import pyplot as plt
from scipy.special import factorial
import numpy as np
from cst_modeling.section import cst_foil_fit
from cst_modeling.math import cst_curve, interp_from_curve, find_circle_3p, clustcos

def fit_curve_custom(x: np.ndarray, y: np.ndarray, n_en=3, n_cst=20, xn1=0.5, xn2=1.0):
    '''
    使用最小二乘法拟合CST曲线
    
    参数:
    -----------
    x: ndarray
        x坐标数组
    y: ndarray
        y坐标数组
    n_en: int
        考虑的参数数量
    n_cst: int
        CST参数阶数
    xn1, xn2: float
        CST形状参数
        
    返回:
    --------
    ndarray
        拟合的CST参数
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

def enhanced_le_cst_fit(xu: np.ndarray, yu: np.ndarray, xl: np.ndarray, yl: np.ndarray, 
                      n_base=7, n_diff=20, xn1=0.5, xn2=1.0, n_enhanced=3):
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
    n_enhanced: int
        增强参数的数量（默认为3）
        
    返回:
    --------
    cst_base_u, cst_base_l: ndarray
        基础翼型的上下表面CST参数
    cst_le_u, cst_le_l: ndarray
        前缘修型的上下表面CST参数（仅前n_enhanced个）
    '''
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
    # 1. 获取基础翼型的CST参数
    cst_base_u, cst_base_l = cst_foil_fit(xu, yu, xl, yl, n_cst=n_base, xn1=xn1, xn2=xn2)
    
    # 2. 使用基础CST参数生成基础翼型
    nn1 = xu.shape[0]
    _, yu_base = cst_curve(nn1, cst_base_u, x=xu, xn1=xn1, xn2=xn2)
    nn2 = xl.shape[0]
    _, yl_base = cst_curve(nn2, cst_base_l, x=xl, xn1=xn1, xn2=xn2)
    
    # 3. 计算差异曲线
    yu_diff = yu - yu_base
    yl_diff = yl - yl_base
    
    # 4. 拟合差异曲线
    cst_diff_u = fit_curve_custom(xu, yu_diff, n_en=n_enhanced, n_cst=n_diff, xn1=xn1, xn2=xn2)
    cst_diff_l = fit_curve_custom(xl, yl_diff, n_en=n_enhanced, n_cst=n_diff, xn1=xn1, xn2=xn2)
    
    # 5. 仅取差异模型的前n_enhanced个参数作为前缘修型参数
    cst_le_u = cst_diff_u[:n_enhanced]
    cst_le_l = cst_diff_l[:n_enhanced]
    
    return cst_base_u, cst_base_l, cst_le_u, cst_le_l

def enhanced_cst_foil(nn: int, cst_base_u, cst_base_l, cst_le_u=None, cst_le_l=None, 
                    x=None, t=None, tail=0.0, xn1=0.5, xn2=1.0):
    '''
    使用CST方法构建具有增强前缘拟合能力的翼型
    
    基于基础翼型与前缘修型叠加的方式构建翼型：
    1. 使用n_base阶伯恩斯坦多项式构造基础翼型
    2. 使用n_diff阶伯恩斯坦多项式构造前缘修型的叠加部分
    3. 在叠加部分，仅前n_enhanced个参数有效，其余参数为0
    
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
    '''
    from numpy import zeros, array, argmax

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

def enhanced_cst_foil_with_fit(xu: np.ndarray, yu: np.ndarray, xl: np.ndarray, yl: np.ndarray,
                             nn=101, t=None, tail=0.0, xn1=0.5, xn2=1.0, n_base=7, n_diff=20, n_enhanced=3):
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
    n_base: int
        基础翼型的CST参数阶数（默认为7阶伯恩斯坦多项式）
    n_diff: int
        差异曲线的CST参数阶数（默认为20阶伯恩斯坦多项式）
    n_enhanced: int
        增强参数的数量（默认为3）
        
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
        xu, yu, xl, yl, n_base=n_base, n_diff=n_diff, xn1=xn1, xn2=xn2, n_enhanced=n_enhanced)
    
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

if __name__ == "__main__":

    cst_base_u = np.array([0.5012099146842957, 0.6857459545135498, -0.4261115789413452, 3.9740867614746094, -6.016767978668213, 10.717917442321777, -10.333446502685547, 10.546847343444824, -5.772887229919434, 4.124373435974121])
    cst_base_l = np.array([-0.33657747507095337, -1.0892202854156494, 1.4455721378326416, -4.642240047454834, 7.1040449142456055, -10.644145011901855, 10.836443901062012, -9.559629440307617, 5.909415245056152, -3.233139753341675])
    cst_le_u = np.array([0.3435426652431488, -0.6757846474647522, 0.6277846693992615])
    cst_le_l = np.array([-0.0510740727186203, 0.25669166445732117, -0.26311492919921875])

    x, yu_new, yl_new, _, _= enhanced_cst_foil(nn = 101,
        cst_base_u = cst_base_u,
        cst_base_l = cst_base_l,
        cst_le_u = cst_le_u,
        cst_le_l = cst_le_l,
        x = None,
        t = None,
        tail = 0.0, xn1=0.5, xn2=1.0)
    
    fig, ax = plt.subplots()
    ax.plot(x, yu_new, label='Upper surface')
    ax.plot(x, yl_new, label='Lower surface')
    ax.legend()
    plt.show()