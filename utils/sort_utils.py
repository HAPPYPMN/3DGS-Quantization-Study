import torch

class QSU:
    def __init__(self, pivots):
        # 将 pivots 转换为 GPU tensor
        self.pivots = pivots.cuda() if not pivots.is_cuda else pivots
    
    def quick_sort(self, tile_depths):
        num_subsets = len(self.pivots) + 1
        subsets_values = [torch.tensor([], device="cuda:0") for _ in range(num_subsets)]
        subsets_indices = [torch.tensor([], device="cuda:0") for _ in range(num_subsets)]
        
        # 将 tile_depths 的维度扩展为 (len(tile_depths), len(pivots))，以便与 pivots 进行比较
        expanded_depths = tile_depths.unsqueeze(1).expand(-1, len(self.pivots))

        # 使用广播比较 tile_depths 与 pivots，得到一个布尔矩阵
        # 每行表示对应 depth 与所有 pivot 的比较结果
        mask = expanded_depths > self.pivots

        # 计算每个 depth 应该放入的 subset 索引
        subset_indices = mask.sum(dim=1)

        # 使用 scatter 将每个 depth 分配到对应的子集，并记录原始索引
        for i in range(num_subsets):
            indices = (subset_indices == i).nonzero(as_tuple=True)[0]
            if indices.numel() > 0:  # 如果该子集中有元素

                subsets_values[i] = tile_depths[indices]
                subsets_indices[i] = indices
        
        return subsets_values, subsets_indices

def my_sort(tile_depths):
    pivots = tile_depths[torch.randperm(len(tile_depths))[:7]]

    qsu = QSU(pivots = pivots)

    # QSU 近似排序
    subsets_values, subsets_indices = qsu.quick_sort(tile_depths)
    # 输出结果，每个 subset 换行显示

    # 在 QSU 外部对每个子集进行排序
    sorted_subsets_values = []
    sorted_subsets_indices = []

    for values, indices in zip(subsets_values, subsets_indices):
        # 对子集的深度值进行排序
        sorted_values, sorted_order = torch.sort(values)
        sorted_indices = indices[sorted_order]

        # 将排序后的值和索引加入新的列表
        sorted_subsets_values.append(sorted_values)
        sorted_subsets_indices.append(sorted_indices)

    all_sorted_values = torch.cat(sorted_subsets_values)
    all_sorted_indices = torch.cat(sorted_subsets_indices).to(torch.int32)
    
    return all_sorted_values, all_sorted_indices
