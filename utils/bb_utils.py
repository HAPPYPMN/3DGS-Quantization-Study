import torch

@torch.no_grad()
def get_radius(cov2d, det):
    mid = 0.5 * (cov2d[:, 0,0] + cov2d[:,1,1])
    lambda1 = mid + torch.sqrt((mid**2-det).clip(min=0.1))
    lambda2 = mid - torch.sqrt((mid**2-det).clip(min=0.1))
    long_radius = (3.0 * torch.sqrt(torch.max(lambda1, lambda2))).ceil()
    short_radius = (3.0 * torch.sqrt(torch.min(lambda1, lambda2))).ceil()
    r_mask = (long_radius > 0) & (det > 0)
    return long_radius, short_radius, r_mask

@torch.no_grad()
def get_AABB(point_images, radii, grid_x, grid_y, block_x, block_y):
    # 计算 rect_min 和 rect_max
    device = point_images.device
    block_tensor = torch.tensor([block_x, block_y], dtype=torch.int32, device=device)
    block_offset = torch.tensor([block_x - 1, block_y - 1], dtype=torch.int32, device=device)
    
    rect_min = (point_images - radii[:, None]) / block_tensor
    rect_max = (point_images + radii[:, None] + block_offset) / block_tensor
    
    # 转换为整型（等效于 int 操作）
    rect_min = rect_min.floor().to(torch.int32)
    rect_max = rect_max.floor().to(torch.int32)
    
    # Clipping to grid boundaries
    rect_min[..., 0] = rect_min[..., 0].clip(0, grid_x)
    rect_min[..., 1] = rect_min[..., 1].clip(0, grid_y)
    rect_max[..., 0] = rect_max[..., 0].clip(0, grid_x)
    rect_max[..., 1] = rect_max[..., 1].clip(0, grid_y)
    
    b_mask = ((rect_max[..., 0] - rect_min[..., 0]) * (rect_max[..., 1] - rect_min[..., 1]) > 0)
    
    return rect_min, rect_max, b_mask

def SAT_subtile(
    means2D:      torch.Tensor,  # (P, 2)
    long_radius:  torch.Tensor,  # (P,)
    short_radius: torch.Tensor,  # (P,)
    cov2d:        torch.Tensor,  # (P, 2, 2)
    grid_cells:   torch.Tensor,   # (N, 2)
    block_x,
    block_y
) -> torch.Tensor:
    """
    批量判断多个椭圆与多个网格单元是否相交，网格细分为 4 个子单元。
    最终返回形状 (P, N, 4)：表示第 i 个椭圆，和第 j 个网格的“4 个子单元”是否分别相交。

    :param means2D:     (P, 2)   - 椭圆中心
    :param long_radius: (P,)     - 椭圆长轴半径
    :param short_radius:(P,)     - 椭圆短轴半径
    :param cov2d:       (P, 2,2) - 椭圆协方差(仅用其特征向量获取旋转方向)
    :param grid_cells:  (N, 2)   - 网格左上角索引
    :return:            (P, N, 4) 布尔张量，表示是否相交
    """

    device = means2D.device

    # 1) ---------------------------------------------------------
    #   计算所有“子单元”的左上角坐标
    #   每个网格 cell 被细分为 4 个 sub-block:
    #     sub_block_x = BLOCK_X / 2
    #     sub_block_y = BLOCK_Y / 2
    #   => 4 个偏移: (0,0), (sub_block_x,0), (0,sub_block_y), (sub_block_x, sub_block_y)
    #
    sub_block_x = block_x / 2.0
    sub_block_y = block_y / 2.0

    sub_offsets = torch.tensor([
        [0,           0          ],  # 左上
        [sub_block_x, 0          ],  # 右上
        [0,           sub_block_y],  # 左下
        [sub_block_x, sub_block_y]   # 右下
    ], device=device)  # (4,2)

    # grid_cells (N,2) => 网格左上角 + 偏移 => 4*N 个子单元左上角
    # 先把 grid_cells => (N,1,2) 乘以 (BLOCK_X, BLOCK_Y)
    # 再加上 sub_offsets => (1,4,2) 广播
    block_xy = torch.tensor([block_x, block_y], device=device)
    grid_cells_world = grid_cells * block_xy  # (N,2)，实际物理坐标(左上角)

    # broadcast => (N,1,2) + (1,4,2) => (N,4,2)
    all_sub_cells = grid_cells_world.unsqueeze(1) + sub_offsets.unsqueeze(0)  # (N,4,2)

    # 把 (N,4,2) view 到 (4*N,2)，其中 4*N 表示“全部子单元”
    all_sub_cells = all_sub_cells.reshape(-1,2)  # (4*N,2)

    # 2) ---------------------------------------------------------
    #   对每个子单元，再构造“4 个顶点” => shape (4*N,4,2)
    #   类似“左上 => 右上 => 右下 => 左下”
    #
    #   左上: all_sub_cells
    #   右上: + (sub_block_x, 0)
    #   右下: + (sub_block_x, sub_block_y)
    #   左下: + (0, sub_block_y)
    #
    sub_vertices = torch.stack([
        all_sub_cells, 
        all_sub_cells + torch.tensor([sub_block_x, 0.0], device=device),
        all_sub_cells + torch.tensor([sub_block_x, sub_block_y], device=device),
        all_sub_cells + torch.tensor([0.0,         sub_block_y], device=device)
    ], dim=1)  # (4*N,4,2)

    # 3) ---------------------------------------------------------
    #   特征分解 + 排序，以确定“主轴向量(major_vec)”与“副轴向量(minor_vec)”
    #
    eigvals, eigvecs = torch.linalg.eigh(cov2d)  # => (P,2), (P,2,2)
    sorted_idx = eigvals.argsort(dim=1, descending=True)  # (P,2)
    idx0 = sorted_idx[:,0].unsqueeze(-1).unsqueeze(-1).expand(-1,2,1)  # => (P,2,1)
    idx1 = sorted_idx[:,1].unsqueeze(-1).unsqueeze(-1).expand(-1,2,1)  # => (P,2,1)
    major_vec = torch.gather(eigvecs, dim=2, index=idx0).squeeze(-1)   # => (P,2)
    minor_vec = torch.gather(eigvecs, dim=2, index=idx1).squeeze(-1)   # => (P,2)

    # 以长轴半径 * major_vec, 短轴半径 * minor_vec
    major_axis = major_vec * long_radius.unsqueeze(-1)   # (P,2)
    minor_axis = minor_vec * short_radius.unsqueeze(-1)  # (P,2)

    # 4) ---------------------------------------------------------
    #   分离轴: 对每个椭圆 => 4 条轴 => (P,4,2)
    #   包含: (1,0), (0,1), major_axis[i], minor_axis[i]
    #
    grid_axes_2 = torch.tensor([[1.0, 0.0],
                                [0.0, 1.0]],
                                device=device).unsqueeze(0)  
    # => shape (1,2,2)，再 expand => (P,2,2)
    grid_axes_2 = grid_axes_2.expand(means2D.size(0), -1, -1)  # (P,2,2)

    ellipse_axes_2 = torch.stack([major_axis, minor_axis], dim=1)  # (P,2,2)

    # 拼接 => (P,4,2)
    all_axes = torch.cat([grid_axes_2, ellipse_axes_2], dim=1)  # (P,4,2)
    # 归一化
    all_axes = all_axes / (all_axes.norm(dim=2, keepdim=True) + 1e-15)

    # 5) ---------------------------------------------------------
    #   计算“子单元顶点”在 4 条轴上的投影
    #   sub_vertices: (4*N,4,2)
    #   all_axes:     (P,4,2)
    #
    #   想要结果 => (P,4,4*N,4)，再对最后一维(4 顶点)取 min/max => (P,4,4*N)
    #   最终可以 reshape => (P,4,N,4)
    #   但为简洁，我们先 broadcast => (P,4,1,1,2) 与 (1,1,4*N,4,2) 相乘
    #
    axes_for_bcast = all_axes.unsqueeze(2).unsqueeze(3)      # => (P,4,1,1,2)
    sv_for_bcast   = sub_vertices.unsqueeze(0).unsqueeze(0)  # => (1,1,4*N,4,2)

    # => 广播 => (P,4,4*N,4,2), sum(dim=-1) => (P,4,4*N,4)
    sub_proj_all = (axes_for_bcast * sv_for_bcast).sum(dim=-1)  # (P,4,4*N,4)

    # 对 4 个顶点做 min/max => (P,4,4*N)
    sub_min_all = sub_proj_all.min(dim=3).values  # (P,4,4*N)
    sub_max_all = sub_proj_all.max(dim=3).values  # (P,4,4*N)

    # 6) ---------------------------------------------------------
    #   椭圆中心投影 + 椭圆在该轴上的半径
    #
    #   椭圆中心 => (P,4)
    ellipse_center_proj = (means2D.unsqueeze(1) * all_axes).sum(dim=-1)  # (P,4)

    #   椭圆半径 =>  |major_axis dot axis| + |minor_axis dot axis|
    ellipse_two_axes = torch.stack([major_axis, minor_axis], dim=1)  # (P,2,2)
    dot_prod = torch.einsum('pab,pcb->pac', ellipse_two_axes, all_axes)  # (P,2,4)
    ellipse_radii = dot_prod.abs().sum(dim=1)  # (P,4)

    # 扩展到 (P,4,4*N)
    P_, N_ = means2D.size(0), grid_cells.size(0)
    ellipse_center_proj_4N = ellipse_center_proj.unsqueeze(-1).expand(-1, -1, 4*N_)
    ellipse_radii_4N       = ellipse_radii.unsqueeze(-1).expand(-1, -1, 4*N_)

    # 7) ---------------------------------------------------------
    #   分离轴定理: 只要在某条轴能分离 => 不相交
    #
    #   no_overlap_on_axis => (P,4,4*N)
    no_overlap_on_axis = (
        (ellipse_center_proj_4N + ellipse_radii_4N < sub_min_all) |
        (ellipse_center_proj_4N - ellipse_radii_4N > sub_max_all)
    )

    # 如果在任意一条轴分离 => not intersect
    separated = no_overlap_on_axis.any(dim=1)  # (P,4*N)

    # => 相交取反 => (P,4*N)
    intersect_flat = ~separated

    # 8) ---------------------------------------------------------
    #   整理成 (P, N, 4)，每个网格有 4 个子单元
    #
    #   intersect_flat.shape = (P, 4*N)
    #   只需 .view(P,N,4)
    result = intersect_flat.view(P_, N_, 4)

    return result

# @torch.no_grad()
# def SAT(
#     means2D:      torch.Tensor,  # (P,2)   - 每个椭圆的中心坐标
#     long_radius:  torch.Tensor,  # (P,)    - 每个椭圆的长轴半径
#     short_radius: torch.Tensor,  # (P,)    - 每个椭圆的短轴半径
#     cov2d:        torch.Tensor,  # (P,2,2) - 椭圆的协方差矩阵(仅用来获取旋转方向)
#     grid_cells:   torch.Tensor   # (N,2)   - 网格单元的左上角索引
# ) -> torch.Tensor:
#     """
#     使用分离轴定理(SAT)，批量判断 P 个椭圆与 N 个网格单元的相交结果。

#     返回: (P, N) 的布尔张量, True 表示相交, False 表示不相交。
#     """

#     device = means2D.device  # 假设所有输入都在同一个 device 上

#     # ============= 1) 构造网格单元 4 个顶点 =======================
#     # grid_cells (N,2) => 这 N 个格子在世界坐标下的左上角坐标
#     # 网格块大小：BLOCK_X, BLOCK_Y
#     block_xy = torch.tensor([BLOCK_X, BLOCK_Y], device=device)

#     # 每个格子的左上角: (N,2)
#     # 右上角: (N,2)
#     # 右下角: (N,2)
#     # 左下角: (N,2)
#     # 拼成 (N,4,2)
#     grid_vertices = torch.stack([
#         grid_cells * block_xy,
#         grid_cells * block_xy + torch.tensor([BLOCK_X, 0.0], device=device),
#         grid_cells * block_xy + torch.tensor([BLOCK_X, BLOCK_Y], device=device),
#         grid_cells * block_xy + torch.tensor([0.0,       BLOCK_Y], device=device)
#     ], dim=1)  # (N,4,2)

#     # ============= 2) 特征分解 + 排序, 获取主轴方向(major)与副轴方向(minor) =========
#     # cov2d: (P,2,2)
#     eigvals, eigvecs = torch.linalg.eigh(cov2d)  # => eigvals: (P,2), eigvecs: (P,2,2)
#     # 其中 eigvecs[:, :, 0] 对应第 0 个特征值, eigvecs[:, :, 1] 对应第 1 个特征值
#     # 但顺序不定，所以先排序

#     # 对特征值在 dim=1 排序 (descending=True 保证前者大、后者小)
#     sorted_idx = eigvals.argsort(dim=1, descending=True)  # (P,2)
#     # 分别取“最大值”索引、“最小值”索引
#     idx0 = sorted_idx[:, 0].unsqueeze(-1).unsqueeze(-1).expand(-1, 2, 1)  # (P,2,1)
#     idx1 = sorted_idx[:, 1].unsqueeze(-1).unsqueeze(-1).expand(-1, 2, 1)  # (P,2,1)

#     # 拿到最大特征值方向(major_vec): (P,2)
#     major_vec = torch.gather(eigvecs, dim=2, index=idx0).squeeze(-1)  # => (P,2)
#     # 拿到最小特征值方向(minor_vec): (P,2)
#     minor_vec = torch.gather(eigvecs, dim=2, index=idx1).squeeze(-1)  # => (P,2)

#     # ============= 3) 计算每个椭圆的主轴向量(major_axis)和副轴向量(minor_axis) ==========
#     # 按照长轴对应 major_vec, 短轴对应 minor_vec
#     # major_axis/minor_axis 都是“中心 -> 椭圆边界”的半轴向量(长度分别是 long_radius[i], short_radius[i])
#     major_axis = major_vec * long_radius.unsqueeze(-1)   # (P,2)
#     minor_axis = minor_vec * short_radius.unsqueeze(-1)  # (P,2)

#     # ============= 4) 对于每个椭圆, 我们需要 4 条分离轴: =============
#     #    1. (1,0)  - 网格X方向
#     #    2. (0,1)  - 网格Y方向
#     #    3. major_axis[i] (单位化)
#     #    4. minor_axis[i] (单位化)
#     #
#     # 这里以批量方式组织 => all_axes: (P,4,2)

#     # 先构造网格的2条轴, shape = (1,2,2), 然后 expand 到 (P,2,2)
#     grid_axes_2 = torch.tensor([[1.0, 0.0],
#                                 [0.0, 1.0]], 
#                                 device=device).unsqueeze(0)
#     grid_axes_2 = grid_axes_2.expand(means2D.size(0), -1, -1)  # (P,2,2)

#     # 再把 major_axis/minor_axis 也组织成 shape=(P,2,2)
#     ellipse_axes_2 = torch.stack([major_axis, minor_axis], dim=1)  # (P,2,2)

#     # 最终4条轴 => concat => (P,4,2)
#     all_axes = torch.cat([grid_axes_2, ellipse_axes_2], dim=1)  # (P,4,2)

#     # 归一化(以防出现非常大的放缩或数值不稳定):
#     all_axes = all_axes / (all_axes.norm(dim=2, keepdim=True) + 1e-15)  # (P,4,2)

#     # ============= 5) 计算网格顶点在这 4 条轴上的投影区间 =============
#     # grid_vertices: (N,4,2) - 每个网格有4顶点
#     # all_axes:      (P,4,2) - 每个椭圆对应4条分离轴
#     #
#     # 我们想要一个 shape = (P,4,N,4) => 其中:
#     #   P => 椭圆索引
#     #   4 => 4条轴
#     #   N => 网格单元索引
#     #   4 => 网格顶点索引
#     #
#     axes_for_grid = all_axes.unsqueeze(2).unsqueeze(3)  # (P,4,1,1,2)
#     gv_for_bcast  = grid_vertices.unsqueeze(0).unsqueeze(0)  # (1,1,N,4,2)

#     grid_proj_all = (axes_for_grid * gv_for_bcast).sum(dim=-1)  # => (P,4,N,4)

#     grid_min_all = grid_proj_all.min(dim=3).values  # (P,4,N)
#     grid_max_all = grid_proj_all.max(dim=3).values  # (P,4,N)


#     # ============= 6) 椭圆中心的投影 + 椭圆在该轴上的半径 =============
#     # 椭圆中心 means2D: (P,2)
#     # 在 all_axes: (P,4,2) 上的投影 => (P,4)
#     ellipse_center_proj = (means2D.unsqueeze(1) * all_axes).sum(dim=-1)  # (P,4)

#     # 椭圆在每条轴上的半径 = (|major_axis dot axis| + |minor_axis dot axis|)
#     # major_axis, minor_axis: (P,2); all_axes: (P,4,2)
#     # 先堆叠 => (P,2,2), einsum 点乘 => (P,2,4)，再在第二个维度 sum => (P,4)
#     ellipse_two_axes = torch.stack([major_axis, minor_axis], dim=1)  # (P,2,2)
#     dot_prod = torch.einsum('pab,pcb->pac', ellipse_two_axes, all_axes)  # (P,2,4)
#     # 对 axis=1 做绝对值、再 sum
#     ellipse_radii = dot_prod.abs().sum(dim=1)  # (P,4)

#     # 扩展到与 (P,4,N) 匹配
#     ellipse_center_proj_4N = ellipse_center_proj.unsqueeze(-1).expand(-1, -1, grid_cells.size(0))  # (P,4,N)
#     ellipse_radii_4N       = ellipse_radii.unsqueeze(-1).expand(-1, -1, grid_cells.size(0))        # (P,4,N)

#     # ============= 7) 分离轴定理: 任意一条轴分开 => 不相交 =============
#     # 在第 i 条轴上:
#     #   椭圆区间 = [center_proj - radius, center_proj + radius]
#     #   网格区间 = [grid_min_all, grid_max_all]
#     #
#     # 如果出现 (ellipse_max < grid_min) or (ellipse_min > grid_max), 则分离成功 => 不相交
#     #    ellipse_max = center_proj + radius
#     #    ellipse_min = center_proj - radius
#     #
#     # (P,4,N) => any(dim=1) 表示只要在 4 条轴中找到一条能分离 => 整体不相交
#     no_overlap_on_axis = (
#         (ellipse_center_proj_4N + ellipse_radii_4N < grid_min_all) |
#         (ellipse_center_proj_4N - ellipse_radii_4N > grid_max_all)
#     )  # (P,4,N)

#     # 在任意一条轴(沿 axis=1)找到分离 => 不相交
#     separated = no_overlap_on_axis.any(dim=1)  # (P,N)
#     # 取反 => 相交
#     intersect = ~separated  # (P,N)

#     return intersect

def duplicateWithKeys(rect_min, rect_max, grid_x, grid_y, P_view):
    """Calculate tile masks for Gaussian projections."""
    tile_masks = torch.zeros((grid_x, grid_y, P_view), dtype=torch.bool)
    for ty in range(grid_y):
        for tx in range(grid_x):
            over_tl = rect_min[..., 0].clip(min=tx), rect_min[..., 1].clip(min=ty)
            over_br = rect_max[..., 0].clip(max=tx+1), rect_max[..., 1].clip(max=ty+1)
            tile_masks[tx, ty, :] = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1])
    return tile_masks