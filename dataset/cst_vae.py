import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from cst_modeling.section import cst_foil  # 导入生成翼型坐标函数
from cst_modeling.math import cst_curve, interp_from_curve, find_circle_3p, clustcos
from cst_modeling.section import cst_foil_fit
latent_dim = 10           # 潜变量维度，可根据实际情况调整

# =============================================================================
# 定义 VAE 模型：包括编码器和解码器
# =============================================================================
class CST_VAE(nn.Module):
    def __init__(self, cst_dim, latent_dim, hidden_dims=None):
        """
        Args:
            cst_dim: 输入（也即输出）的维度，对应 CST 参数数量
            latent_dim: 潜变量维度
            hidden_dims: 隐藏层配置列表；如果为 None，则采用默认配置
        """
        super(CST_VAE, self).__init__()
        self.cst_dim = cst_dim
        self.latent_dim = latent_dim
        
        if hidden_dims is None:
            hidden_dims = [128, 64]
        
        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Linear(cst_dim, hidden_dims[0]),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LeakyReLU(0.2)
        )
        # 从编码器输出映射到潜变量均值与对数方差（log variance）
        self.fc_mu = nn.Linear(hidden_dims[1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[1], latent_dim)

        # 解码器部分：先将潜变量映射到解码器输入，再经过多层反向映射回原始维度
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[1])
        self.decoder = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dims[0], cst_dim)  # 输出层直接回归 CST 参数
        )

    def reparameterize(self, mu, logvar):
        """
        利用重参数化技巧采样潜变量 z ~ N(mu, exp(logvar))
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        logvar = torch.clamp(logvar, min=-10, max=10)
        z = self.reparameterize(mu, logvar)
        decoder_in = self.decoder_input(z)
        reconstruction = self.decoder(decoder_in)
        return reconstruction, mu, logvar

# =============================================================================
# 定义损失函数：重构误差 + KL 散度
# =============================================================================
def loss_function(reconstruction, x, mu, logvar):
    """
    重构损失采用均方误差（MSE），同时加入 KL 散度作为正则项。
    为了防止数值爆炸，对 logvar 进行了 clamp，并且采用了 mean 作为 reduction。
    """
    # 限制 logvar 的范围，防止指数运算导致数值溢出
    logvar = torch.clamp(logvar, min=-10, max=10)
    # 使用 'mean' 作为 reduction
    recon_loss = nn.functional.mse_loss(reconstruction, x, reduction='mean')
    # KL 散度项同样采用 mean 计算
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld

# =============================================================================
# 加载数据函数，假定数据保存在 "airfoil_cst.npz"，真实参数保存在 "labels" 键下
# =============================================================================
# 修改load_real_data函数以支持增强数据集
def load_enhanced_data(npz_file, use_enhanced=True):
    data = np.load(npz_file)
    if use_enhanced:
        # 增强数据集中有base_u, base_l, le_u, le_l四个数组
        base_u = data["base_u"].astype("float32")
        base_l = data["base_l"].astype("float32")
        le_u = data["le_u"].astype("float32")
        le_l = data["le_l"].astype("float32")
        
        # 合并所有参数为一个大数组
        samples_count = base_u.shape[0]
        params_list = []
        for i in range(samples_count):
            # 按顺序组合所有参数: base_u, le_u, base_l, le_l
            combined = np.concatenate([base_u[i], le_u[i], base_l[i], le_l[i]])
            params_list.append(combined)
        
        real_params = np.array(params_list)
        return real_params
    else:
        # 标准模式已移除
        raise ValueError("Only enhanced CST mode is supported")

# 修改train函数以支持增强数据集
def train(beta1=0.0001, use_enhanced=True, data_file="dataset/airfoil_enhanced_cst.npz"):
    # 训练参数设置
    epochs = 1000             # 训练迭代次数
    batch_size = 32           # 批量大小
    checkpoint_dir = "vae_checkpoints"  # 模型检查点保存目录

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备：", device)

    if not os.path.exists(data_file):
        print(f"数据文件 {data_file} 不存在，请先生成数据集。")
        return

    # 加载数据及构建 DataLoader
    real_params = load_enhanced_data(data_file, use_enhanced=use_enhanced)
    cst_dim = real_params.shape[1]
    print(f"CST参数维度: {cst_dim}")
    dataset = TensorDataset(torch.tensor(real_params))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化 VAE 模型并移动到相应设备
    vae = CST_VAE(cst_dim=cst_dim, latent_dim=latent_dim)
    vae = vae.to(device)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    
    # 定义 lr_scheduler：每50个epoch降低学习率为原来的0.5倍
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_loss = float('inf')

    # 模型训练
    for epoch in range(epochs):
        vae.train()
        train_total_loss = 0.0
        train_total_recon_loss = 0.0
        train_total_kl_loss = 0.0

        # 线性增加 beta，从 0 到 beta1
        beta = min(beta1, epoch / (epochs * 0.5))  # 前50%的epoch从0线性增加到beta1

        with tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False) as pbar:
            for batch in pbar:
                x = batch[0].to(device)
                optimizer.zero_grad()
                reconstruction, mu, logvar = vae(x)

                # 计算重构损失（均方误差）
                recon_loss = nn.functional.mse_loss(reconstruction, x, reduction='mean')
                # 为防止数值爆炸，对 logvar 做 clamp
                logvar = torch.clamp(logvar, min=-10, max=10)
                # 计算 KL 散度
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                # 总 loss
                loss = recon_loss + beta * kl_loss

                loss.backward()
                optimizer.step()

                # 累加各部分 loss
                train_total_loss += loss.item() * x.size(0)
                train_total_recon_loss += recon_loss.item() * x.size(0)
                train_total_kl_loss += kl_loss.item() * x.size(0)

                pbar.set_postfix({
                    "recon_loss": f"{recon_loss.item():.6f}",
                    "kl_loss": f"{kl_loss.item():.6f}",
                    "beta": f"{beta:.6f}",
                    "total_loss": f"{loss.item():.6f}"
                })

        # 每个epoch结束后更新学习率调度器
        scheduler.step()

        # 计算每个 epoch 的平均 loss
        avg_loss = train_total_loss / len(dataset)
        avg_recon_loss = train_total_recon_loss / len(dataset)
        avg_kl_loss = train_total_kl_loss / len(dataset)
        print(f"Epoch {epoch + 1}: Average Total Loss: {avg_loss:.6f}, Recon Loss: {avg_recon_loss:.6f}, KL Loss: {avg_kl_loss:.6f}, beta: {beta:.6f}, lr: {optimizer.param_groups[0]['lr']:.6f}")

        # 每 100 个 epoch 保存一次检查点
        if (epoch + 1) % 100 == 0:
            model_type = "enhanced" if use_enhanced else "standard"
            latest_checkpoint_path = os.path.join(checkpoint_dir, f"{model_type}_beta{beta1}_latest.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": vae.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss
            }, latest_checkpoint_path)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_checkpoint_path = os.path.join(checkpoint_dir, f"{model_type}_beta{beta1}_best.pth")
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": vae.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss
                }, best_checkpoint_path)

    print("训练完成。")

    # 利用训练好的 VAE 生成新的 CST 参数样本
    vae.eval()
    with torch.no_grad():
        # 从标准正态分布采样潜变量 z
        z = torch.randn(5, latent_dim, device=device)
        decoder_in = vae.decoder_input(z)
        generated = vae.decoder(decoder_in)
    print("生成的新 CST 参数样本：")
    print(generated.cpu().numpy())

def plot_airfoils(n_samples=5, 
                  data_file="dataset/airfoil_enhanced_cst.npz", 
                  checkpoint_path="vae_checkpoints/enhanced_beta0.0001_best.pth", 
                  latent_dim=latent_dim,
                  sigma=1.0):
    """
    Plot airfoils from dataset and VAE generated samples
    """
    # 加载增强CST数据集
    data = np.load(data_file)
    base_u = data["base_u"].astype("float32")
    base_l = data["base_l"].astype("float32")
    le_u = data["le_u"].astype("float32")
    le_l = data["le_l"].astype("float32")
    
    n_base_u = base_u.shape[1]
    n_le_u = le_u.shape[1]
    n_base_l = base_l.shape[1]
    n_le_l = le_l.shape[1]
    
    plt.figure(figsize=(10, 6))
    
    # 绘制数据集中的部分翼型（黑色，较细线宽）
    sample_indices = np.random.choice(len(base_u), min(20, len(base_u)), replace=False)
    for idx in sample_indices:
        # 生成翼型坐标 - 基础翼型加修正部分
        x, yu_base, yl_base, _, _ = cst_foil(201, base_u[idx], base_l[idx], x=None, t=None, tail=0.0)
        
        # 创建修正部分的完整参数数组
        n_le = 40  # 修正部分使用40阶多项式
        cst_le_u_full = np.zeros(n_le)
        cst_le_u_full[:n_le_u] = le_u[idx]
        _, yu_le = cst_curve(201, cst_le_u_full, x=x, xn1=0.5, xn2=1.0)
        
        cst_le_l_full = np.zeros(n_le)
        cst_le_l_full[:n_le_l] = le_l[idx]
        _, yl_le = cst_curve(201, cst_le_l_full, x=x, xn1=0.5, xn2=1.0)
        
        # 基础翼型加修正部分
        yu = yu_base + yu_le
        yl = yl_base + yl_le
        
        plt.plot(x, yu, 'k-', linewidth=0.1, alpha=0.5)
        plt.plot(x, yl, 'k-', linewidth=0.1, alpha=0.5)
    
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 合并数据计算总维度
    sample = np.concatenate([base_u[0], le_u[0], base_l[0], le_l[0]])
    cst_dim = len(sample)
    vae = CST_VAE(cst_dim=cst_dim, latent_dim=latent_dim)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vae.load_state_dict(checkpoint["model_state_dict"])
    vae.to(device)
    vae.eval()
    
    # 生成翼型样本
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim, device=device) * sigma
        decoder_in = vae.decoder_input(z)
        generated = vae.decoder(decoder_in)
    
    # 分解并绘制生成的翼型
    colors = ['r', 'g', 'b', 'c', 'm', 'y']  # 只使用单字符颜色代码
    for i in range(n_samples):
        params = generated[i].cpu().numpy()
        # 按顺序分解参数: base_u, le_u, base_l, le_l
        base_u_i = params[:n_base_u]
        le_u_i = params[n_base_u:n_base_u+n_le_u]
        base_l_i = params[n_base_u+n_le_u:n_base_u+n_le_u+n_base_l]
        le_l_i = params[n_base_u+n_le_u+n_base_l:]
        
        # 生成基础翼型
        x, yu_base, yl_base, _, _ = cst_foil(201, base_u_i, base_l_i, x=None, t=None, tail=0.0)
        
        # 生成修正部分
        n_le = 40
        cst_le_u_full = np.zeros(n_le)
        cst_le_u_full[:n_le_u] = le_u_i
        _, yu_le = cst_curve(201, cst_le_u_full, x=x, xn1=0.5, xn2=1.0)
        
        cst_le_l_full = np.zeros(n_le)
        cst_le_l_full[:n_le_l] = le_l_i
        _, yl_le = cst_curve(201, cst_le_l_full, x=x, xn1=0.5, xn2=1.0)
        
        # 合成最终翼型
        yu = yu_base + yu_le
        yl = yl_base + yl_le
        
        color = colors[i % len(colors)]
        plt.plot(x, yu, color=color, linestyle='-', linewidth=0.5)
        plt.plot(x, yl, color=color, linestyle='--', linewidth=0.5)
    
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Airfoil Database and Generated Samples")
    plt.xlim((-0.05, 1.05))
    plt.ylim((-0.3, 0.3))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def generate_enriched_airfoils(vae, latent_dim, n_samples=10, sigma=2.0):
    """
    利用训练好的 VAE 生成更多样化的翼型。这里通过调整采样方差 sigma 来增强多样性。
    
    参数:
        vae: 训练好的 VAE 模型
        latent_dim: 潜变量维度
        n_samples: 生成样本数量
        sigma: 采样时的标准差，默认大于1以增加多样性
    """
    device = next(vae.parameters()).device
    with torch.no_grad():
        # 用 sigma 代替标准正态分布中的标准偏差
        z = torch.randn(n_samples, latent_dim, device=device) * sigma
        decoder_in = vae.decoder_input(z)
        generated = vae.decoder(decoder_in)
    return generated

def plot_enriched_airfoils(n_samples=5, sigma=2.0, 
                           data_file="dataset/airfoil_enhanced_cst.npz", 
                           checkpoint_path="vae_checkpoints/enhanced_beta0.0001_best.pth", 
                           latent_dim=latent_dim):
    """
    Plot diverse airfoils generated by VAE
    """
    # 加载数据获取参数维度信息
    data = np.load(data_file)
    base_u = data["base_u"].astype("float32")
    base_l = data["base_l"].astype("float32")
    le_u = data["le_u"].astype("float32")
    le_l = data["le_l"].astype("float32")
    
    n_base_u = base_u.shape[1]
    n_le_u = le_u.shape[1]
    n_base_l = base_l.shape[1]
    n_le_l = le_l.shape[1]
    cst_dim = n_base_u + n_le_u + n_base_l + n_le_l
    
    plt.figure(figsize=(10, 6))
    
    # 加载VAE模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = CST_VAE(cst_dim=cst_dim, latent_dim=latent_dim)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vae.load_state_dict(checkpoint["model_state_dict"])
    vae.to(device)
    vae.eval()
    
    # 生成增强翼型
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim, device=device) * sigma
        decoder_in = vae.decoder_input(z)
        generated = vae.decoder(decoder_in)
    
    # 定义颜色列表
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    
    # 绘制生成的翼型，每个翼型使用不同颜色
    for i in range(n_samples):
        params = generated[i].cpu().numpy()
        # 按顺序分解参数: base_u, le_u, base_l, le_l
        base_u_i = params[:n_base_u]
        le_u_i = params[n_base_u:n_base_u+n_le_u]
        base_l_i = params[n_base_u+n_le_u:n_base_u+n_le_u+n_base_l]
        le_l_i = params[n_base_u+n_le_u+n_base_l:]
        
        # 生成基础翼型
        x, yu_base, yl_base, _, _ = cst_foil(61, base_u_i, base_l_i, x=None, t=None, tail=0.0)
        
        # 生成修正部分
        n_le = 40
        cst_le_u_full = np.zeros(n_le)
        cst_le_u_full[:n_le_u] = le_u_i
        _, yu_le = cst_curve(61, cst_le_u_full, x=x, xn1=0.5, xn2=1.0)
        
        cst_le_l_full = np.zeros(n_le)
        cst_le_l_full[:n_le_l] = le_l_i
        _, yl_le = cst_curve(61, cst_le_l_full, x=x, xn1=0.5, xn2=1.0)
        
        # 合成最终翼型
        yu = yu_base + yu_le
        yl = yl_base + yl_le
        
        color = colors[i % len(colors)]
        plt.plot(x, yu, f'{color}-', linewidth=0.7, label=f'Airfoil {i+1} Upper')
        plt.plot(x, yl, f'{color}--', linewidth=0.7, label=f'Airfoil {i+1} Lower')
    
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Diverse Airfoil Generation")
    plt.xlim((-0.05, 1.05))
    plt.ylim((-0.15, 0.15))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def generate_variants_for_sample(vae, sample, num_variants=5, noise_std=0.1):
    """
    利用训练好的 VAE 为给定翼型生成多个变种。

    参数:
        vae: 训练好的 VAE 模型
        sample: 给定翼型的 CST 参数（Tensor，形状为 [1, cst_dim]）
        num_variants: 要生成的变种数量
        noise_std: 在潜空间中添加噪声的标准差

    返回:
        variants: 生成的翼型参数变种列表，每个元素形状为 [1, cst_dim]
    """
    device = next(vae.parameters()).device
    sample = sample.to(device)
    vae.eval()
    
    with torch.no_grad():
        # 编码得到潜空间表示
        encoded = vae.encoder(sample)
        mu = vae.fc_mu(encoded)
        # 使用 mu 作为基础，在其上添加噪声生成变种
        variants = []
        for _ in range(num_variants):
            noise = torch.randn_like(mu) * noise_std
            z_variant = mu + noise
            decoder_in = vae.decoder_input(z_variant)
            variant_params = vae.decoder(decoder_in)
            variants.append(variant_params)
    
    return variants

def plot_airfoil_variants(sample, variants, title="Original and Variants"):
    """
    Plot original airfoil and its variants with improved visualization
    """
    plt.figure(figsize=(10, 4))
    
    # Load data to get parameter dimensions
    data = np.load("CLIP_finale/train_dataset/airfoil_enhanced_cst.npz")
    base_u = data["base_u"].astype("float32")
    base_l = data["base_l"].astype("float32")
    le_u = data["le_u"].astype("float32")
    le_l = data["le_l"].astype("float32")
    
    n_base_u = base_u.shape[1]
    n_le_u = le_u.shape[1]
    n_base_l = base_l.shape[1]
    n_le_l = le_l.shape[1]
    
    # Parse original airfoil parameters
    base_u_s = sample[:n_base_u]
    le_u_s = sample[n_base_u:n_base_u+n_le_u]
    base_l_s = sample[n_base_u+n_le_u:n_base_u+n_le_u+n_base_l]
    le_l_s = sample[n_base_u+n_le_u+n_base_l:]
    
    # Generate base airfoil
    x, yu_base, yl_base, _, _ = cst_foil(201, base_u_s, base_l_s, x=None, t=None, tail=0.0)
    
    # Generate correction part
    n_le = 40
    cst_le_u_full = np.zeros(n_le)
    cst_le_u_full[:n_le_u] = le_u_s
    _, yu_le = cst_curve(201, cst_le_u_full, x=x, xn1=0.5, xn2=1.0)
    
    cst_le_l_full = np.zeros(n_le)
    cst_le_l_full[:n_le_l] = le_l_s
    _, yl_le = cst_curve(201, cst_le_l_full, x=x, xn1=0.5, xn2=1.0)
    
    # Composite final airfoil
    yu = yu_base + yu_le
    yl = yl_base + yl_le
    
    # Plot original airfoil (black lines)
    plt.plot(x, yu, color='k', linestyle='-', linewidth=1.5, label="Original")
    plt.plot(x, yl, color='k', linestyle='-', linewidth=1.5)
    
    # Use better color scheme
    try:
        import seaborn as sns
        colors = sns.color_palette("bright", len(variants))
    except ImportError:
        # If seaborn not available, use better predefined colors
        colors = ['#FF4500', '#1E90FF', '#32CD32', '#9932CC', '#FF8C00', 
                 '#008B8B', '#FF1493', '#4682B4', '#FF6347', '#008000']
    
    # Create custom legend handlers
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='k', lw=1.5, label='Original')]
    
    n_variants = len(variants)
    for i in range(n_variants):
        # Convert tensor to numpy array if needed
        variant_params = variants[i]
        if isinstance(variant_params, torch.Tensor):
            variant_params = variant_params.cpu().numpy().flatten()
        
        # Parse parameters
        base_u_i = variant_params[:n_base_u]
        le_u_i = variant_params[n_base_u:n_base_u+n_le_u]
        base_l_i = variant_params[n_base_u+n_le_u:n_base_u+n_le_u+n_base_l]
        le_l_i = variant_params[n_base_u+n_le_u+n_base_l:]
        
        # Generate airfoil
        x, yu_base, yl_base, _, _ = cst_foil(201, base_u_i, base_l_i, x=None, t=None, tail=0.0)
        
        # Generate correction part
        cst_le_u_full = np.zeros(n_le)
        cst_le_u_full[:n_le_u] = le_u_i
        _, yu_le = cst_curve(201, cst_le_u_full, x=x, xn1=0.5, xn2=1.0)
        
        cst_le_l_full = np.zeros(n_le)
        cst_le_l_full[:n_le_l] = le_l_i
        _, yl_le = cst_curve(201, cst_le_l_full, x=x, xn1=0.5, xn2=1.0)
        
        # Composite airfoil
        yu_variant = yu_base + yu_le
        yl_variant = yl_base + yl_le
        
        color = colors[i % len(colors)]
        # Plot variant airfoil, but don't add to legend for every part
        plt.plot(x, yu_variant, linestyle='-', linewidth=1.0, color=color)
        plt.plot(x, yl_variant, linestyle='--', linewidth=1.0, color=color)
        
        # Add custom legend element
        legend_elements.append(Line2D([0], [0], color=color, lw=1.0, 
                                     label=f'Variant {i+1}'))
    
    # Set chart properties
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlim((-0.05, 1.05))
    plt.ylim((-0.15, 0.15))
    plt.grid(True, alpha=0.3)
    
    # Add custom legend
    plt.legend(handles=legend_elements, fontsize=14, loc='best', 
              framealpha=0.7, ncol=2)
    
    # Add note about line styles
    #plt.figtext(0.02, 0.02, "Solid: Upper Surface  Dashed: Lower Surface", fontsize=9)
    
    plt.tight_layout()
    plt.show()

def sample(num_variants=5, noise_std=0.15):
    # 配置
    data_file = "CLIP_finale/train_dataset/airfoil_enhanced_cst.npz"
    checkpoint_path = "CLIP_finale/vae_checkpoints/enhanced_beta0.0001_best.pth"
    
    # 检查文件是否存在
    if not os.path.exists(data_file):
        print(f"Data file {data_file} does not exist. Please check the path or generate the dataset first.")
        return

    # 加载增强数据集
    data = np.load(data_file)
    base_u = data["base_u"].astype("float32")
    base_l = data["base_l"].astype("float32")
    le_u = data["le_u"].astype("float32")
    le_l = data["le_l"].astype("float32")
    
    # 获取一个样本
    idx = np.random.randint(0, len(base_u))
    sample_combined = np.concatenate([base_u[idx], le_u[idx], base_l[idx], le_l[idx]])
    #base_u = np.array([0.11381156017186894, 0.24448629029821423, -0.00828263763189227, 0.4527350683092132, -0.08027724930070027, 0.3897322998268202, 0.09020802212766173, 0.249772768630484, 0.19644220324486403, 0.25162179218041125])
    #base_l= np.array([-0.10281231186000653, -0.06560447837183736, 0.0901022239476216, -0.26296503987383996, 0.46327320395832894, -0.4535718936968189, 0.416942268917615, -0.10520028458211086, 0.11002303888312842, 0.06556575427889524])
    #le_u= np.array([-0.029146489353207922, 0.05987475329744009, -0.05737082671929496])
    #le_l= np.array([-0.03852354996936907, 0.07918431507115493, -0.07848947296456174])
    #sample_combined = np.concatenate([base_u, le_u, base_l, le_l])
    sample = torch.tensor(sample_combined, dtype=torch.float32).view(1, -1)
    sample_np = sample.numpy().flatten()

    # 初始化VAE模型
    cst_dim = sample_np.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = CST_VAE(cst_dim=cst_dim, latent_dim=latent_dim)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vae.load_state_dict(checkpoint["model_state_dict"])
    vae = vae.to(device)

    # 生成变种
    variants = generate_variants_for_sample(vae, sample, num_variants=num_variants, noise_std=noise_std)

    # 绘制原始翼型及其变种
    plot_airfoil_variants(sample_np, variants, title=f"Airfoil with Variants")

if __name__ == "__main__":
    # 使用增强的CST数据集训练
    #train(beta1=0.00001, use_enhanced=True, data_file="dataset/airfoil_enhanced_cst.npz")
    
    #train(beta1=0.0001)
    #plot_airfoils(n_samples=10, sigma=1, checkpoint_path="vae_checkpoints/enhanced_beta0.0001_best.pth")
    #plot_enriched_airfoils(n_samples=10, sigma=2, checkpoint_path="vae_checkpoints/best_0.01.pth")
    sample(num_variants=5, noise_std=0.15)