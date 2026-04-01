import os

import numpy as np
import pandas as pd
import scanpy as sc

import sklearn.neighbors

from collections import defaultdict

from datasets.local_graph_utils import build_local_graph_metadata


# return the neighborhood nodes 
def Cal_Spatial_Net_row_col(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True):
    """
    根据 spot 的 array_row 和 array_col 坐标构建空间邻接图。

    参数:
        adata: AnnData 对象，要求 adata.obs 中包含 `array_row` 和 `array_col`。
        rad_cutoff: 当 model='Radius' 时使用的半径阈值。
        k_cutoff: 当 model='KNN' 时使用的近邻个数。
        model: 邻接图构建方式，支持 'Radius' 或 'KNN'。
        verbose: 是否打印建图信息。

    返回:
        temp_dic: 以 "row_col" 为键的邻接字典，值为相邻 spot 的 "row_col" 列表。
    """

    assert(model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')

    # 取出每个 spot 的二维空间坐标，并整理成后续近邻搜索所需的 DataFrame。
    coor = pd.DataFrame(np.stack([adata.obs['array_row'], adata.obs['array_col']], axis=1))

    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    if model == 'Radius':
        # 基于半径搜索邻居，返回每个点在给定半径内的所有邻居及距离。
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):      #indices是邻居索引列表
            # 每一行记录: 当前点索引、邻居点索引、两者距离。
            KNN_list.append(pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])))
    
    if model == 'KNN':
        # KNN 会把点自身也算进最近邻，因此这里使用 k_cutoff + 1。
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff+1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices.shape[1],indices[it,:], distances[it,:])))

    # 合并所有点的邻接结果，并统一列名。
    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']     #Cell1是中心细胞，Cell2是邻居细胞， Distance是距离

    Spatial_Net = KNN_df.copy()
    # 去掉自己到自己的连边，仅保留真实邻居关系。
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance']>0,]
    # 将整数索引映射回 adata.obs 中原始的 spot 名称。
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' %(Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' %(Spatial_Net.shape[0]/adata.n_obs))

    # 将边表保存到 AnnData 中，便于后续复用。
    adata.uns['Spatial_Net'] = Spatial_Net

    # 构建 "row_col" -> 邻接 "row_col" 列表的字典格式输出。
    temp_dic = defaultdict(list)
    for i in range(Spatial_Net.shape[0]):

        center = Spatial_Net.iloc[i, 0]
        side = Spatial_Net.iloc[i, 1]

        # 用 array_row 和 array_col 拼接出 spot 的二维坐标名称。
        center_name = str(adata.obs['array_row'][center]) + '_' + str(adata.obs['array_col'][center])
        side_name = str(adata.obs['array_row'][side]) + '_' + str(adata.obs['array_col'][side])

        temp_dic[center_name].append(side_name)

    return temp_dic


class SMA(object):
    def __init__(self, path_img, rna_path, msi_path, n_top_genes=3000, n_top_targets=50):
        
        training_slides = ['V11L12-109_B1', 'V11L12-109_C1']
        testing_slides = ['V11L12-109_A1']

        self.path_img = path_img

        rna_adata_list, msi_adata_list = [], []
        rna_highly_variable_list, msi_highly_variable_list = [], []

        for slide in training_slides + testing_slides:

            adata_rna = sc.read_visium(os.path.join(rna_path, slide))
            adata_rna.var_names_make_unique()

            # 对每张切片单独做 HVG 筛选和标准 scRNA 预处理。
            # 后面会对 3 张切片的 HVG 结果取交集，只保留跨切片都稳定的基因。
            sc.pp.highly_variable_genes(adata_rna, flavor="seurat_v3", n_top_genes=n_top_genes)
            sc.pp.normalize_total(adata_rna, target_sum=1e4)
            sc.pp.log1p(adata_rna)

            rna_adata_list.append(adata_rna)
            rna_highly_variable_list.append(adata_rna.var['highly_variable'].values)

            adata_msi = sc.read_h5ad(os.path.join(msi_path, 'metabolite_' + slide + '.h5ad'))
            adata_msi.var_names_make_unique()
            # MSI 也按切片独立筛高变代谢物，后面同样取交集；
            # 这里只做 log1p，不再做 total-count normalize。
            sc.pp.highly_variable_genes(adata_msi, flavor="seurat_v3", n_top_genes=n_top_targets)
            sc.pp.log1p(adata_msi)

            msi_adata_list.append(adata_msi)
            msi_highly_variable_list.append(adata_msi.var['highly_variable'].values)


        ##############
        # 预先统计所有切片 RNA 的全局均值/方差，便于后续做基因级标准化。
        # 当前这个类里并没有实际用到这两个量，属于保留的统计信息。
        temp = np.concatenate([ rna_adata_list[0].X.toarray(), rna_adata_list[1].X.toarray(), rna_adata_list[2].X.toarray()], axis=0)
        self.rna_mean, self.rna_std = temp.mean(axis=0)[None, ], temp.std(axis=0)[None, ]
        
        temp_mask = (self.rna_std == 0)
        # 避免后续若做 z-score 标准化时出现除以 0。
        self.rna_std[temp_mask] = 1

        ###############

        # 只保留 3 张切片共同的高变基因/高变代谢物，
        # 这样训练和测试落在同一套稳定特征空间里。
        self.rna_mask =  rna_highly_variable_list[0] & rna_highly_variable_list[1] & rna_highly_variable_list[2]
        self.msi_mask = msi_highly_variable_list[0] & msi_highly_variable_list[1] & msi_highly_variable_list[2]

        self.training = self._process_data(rna_adata_list[0:2], msi_adata_list[0:2], training_slides)         #前两个切片
        self.testing = self._process_data(rna_adata_list[2:], msi_adata_list[2:], testing_slides)             #测试用切片

        self.rna_length = (self.rna_mask * 1).sum()
        self.msi_length = (self.msi_mask * 1).sum()
        # source_panel/target_panel 保存模型真实使用的输入基因名和输出代谢物名，
        # 训练后的可视化和 attribution 分析都会依赖这两个索引。
        self.target_panel = adata_msi.var['metabolism'].values[self.msi_mask].tolist()  #[i for i in range(self.msi_length)]
        self.source_panel = adata_rna.var_names[self.rna_mask]

        num_training_spots, num_testing_spots = len(self.training), len(self.testing)
        num_training_slides, num_testing_slides = len(training_slides), len(testing_slides)

        ori_num_training_spots = rna_adata_list[0].shape[0] + rna_adata_list[1].shape[0]
        ori_num_testing_spots = rna_adata_list[2].shape[0]

        print("=> SMA loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # num | ")
        print("  ------------------------------")
        print("  train    |  Without filtering {:5d} spots from {:5d} slides ".format(ori_num_training_spots, num_training_slides))
        print("  test     |  Without filtering {:5d} spots from {:5d} slides".format(ori_num_testing_spots, num_testing_slides))
        print("  train    |  After filting {:5d} spots from {:5d} slides ".format(num_training_spots, num_training_slides))
        print("  test     |  After filting {:5d} spots from {:5d} slides".format(num_testing_spots, num_testing_slides))
        print("  ------------------------------")


    def _dictionary_data(self, adata, rna=False):
        dictionary = {}
        array_row, array_col = adata.obs['array_row'].values, adata.obs['array_col'].values

        array = adata.X.toarray()
    
        for i in range(adata.shape[0]):
            # 用 "row_col" 作为统一坐标键，便于 RNA、MSI 和邻接图按空间位置对齐。
            dictionary[str(int(array_row[i])) + '_' +  str(int(array_col[i])) ] = array[i]

        return dictionary

        
    def _process_data(self, rna_adata_list, msi_adata_list, names):

        dataset = []

        for i in range(len(rna_adata_list)):

            rna_temp_adata = rna_adata_list[i]
            msi_temp_adata = msi_adata_list[i]

            rna_dic = self._dictionary_data(rna_temp_adata, rna=True)
            msi_dic = self._dictionary_data(msi_temp_adata)

            # graph = Cal_Spatial_Net_spatial_cite_seq(rna_temp_adata,  k_cutoff=8, model='KNN')  
            graph_1 = Cal_Spatial_Net_row_col(rna_temp_adata,  rad_cutoff=2**(1/2), model='Radius')    #为什么是根号2？这样不是会选到一共8个细胞吗？
            graph_2 = Cal_Spatial_Net_row_col(rna_temp_adata,  rad_cutoff=2, model='Radius')

            rna_keys, msi_keys = rna_dic.keys(), msi_dic.keys()

            for key in rna_keys:
                # 只保留 RNA 与 MSI 都存在的坐标，保证监督信号是空间对齐的。
                if key not in msi_keys:
                    continue
                else:
                    # 中心 spot 仅保留共同高变特征，作为模型的输入/目标。
                    rna_temp = rna_dic[key][self.rna_mask]
                    msi_temp = msi_dic[key][self.msi_mask]

                    # quality check for rna and msi
                    # 若筛完后该 spot 在任一模态上全 0，则认为没有有效信息。
                    if rna_temp.sum() == 0 or msi_temp.sum() == 0:
                        continue
                    else:
                        # 每个空间位置还会关联一张 histology patch，形成图像 + 组学联合输入。
                        img_path = os.path.join(self.path_img, names[i], key + '.png')
                        center_coord = np.array([float(item) for item in key.split('_')], dtype=np.float32)

                        rna_neighbors, msi_neighbors = [], []
                        neighbor_coords, hop_ids, valid_neighbor_mask = [], [], []

                        neighbors_1, neighbors_2 = graph_1[key], graph_2[key]
                        # 第二圈邻居去掉已被第一圈覆盖的点，避免重复。
                        neighbors_2 = [item for item in neighbors_2 if item not in neighbors_1]

                        # connect to the first round 
                        for j in neighbors_1:
                            # 邻居缺失时补零向量，保持固定输入维度。
                            if j not in rna_keys:
                                rna_neighbors.append(np.zeros_like(rna_temp))
                                neighbor_coords.append(center_coord.copy())
                                valid_neighbor_mask.append(False)
                            else:
                                rna_neighbors.append(rna_dic[j][self.rna_mask])
                                neighbor_coords.append(np.array([float(item) for item in j.split('_')], dtype=np.float32))
                                valid_neighbor_mask.append(True)
                            hop_ids.append(1)

                            if j not in msi_keys:
                                msi_neighbors.append(np.zeros_like(msi_temp))
                            else:
                                msi_neighbors.append(msi_dic[j][self.msi_mask])

                        # 第一圈固定保留 4 个槽位，不足则补零。
                        if len(neighbors_1) != 4:
                            for _ in range(4-len(neighbors_1)):
                                rna_neighbors.append(np.zeros_like(rna_temp))
                                msi_neighbors.append(np.zeros_like(msi_temp))
                                neighbor_coords.append(center_coord.copy())
                                hop_ids.append(1)
                                valid_neighbor_mask.append(False)

                        # connect to the second round
                        for j in neighbors_2:
                            # 第二圈与第一圈相同，也使用固定长度的邻域编码。
                            if j not in rna_keys:
                                rna_neighbors.append(np.zeros_like(rna_temp))
                                neighbor_coords.append(center_coord.copy())
                                valid_neighbor_mask.append(False)
                            else:
                                rna_neighbors.append(rna_dic[j][self.rna_mask])
                                neighbor_coords.append(np.array([float(item) for item in j.split('_')], dtype=np.float32))
                                valid_neighbor_mask.append(True)
                            hop_ids.append(2)

                            if j not in msi_keys:
                                msi_neighbors.append(np.zeros_like(msi_temp))
                            else:
                                msi_neighbors.append(msi_dic[j][self.msi_mask])

                        # 第二圈同样固定为 4 个槽位，因此总邻域大小是 8。
                        if len(neighbors_2) != 4:
                            for _ in range(4-len(neighbors_2)):
                                rna_neighbors.append(np.zeros_like(rna_temp))
                                msi_neighbors.append(np.zeros_like(msi_temp))
                                neighbor_coords.append(center_coord.copy())
                                hop_ids.append(2)
                                valid_neighbor_mask.append(False)

                        # 最终一个样本由：
                        # 中心图像 + 中心 RNA + 中心 MSI + 8 个 RNA 邻居 + 8 个 MSI 邻居 + sample id 组成。
                        rna_neighbors = np.stack(rna_neighbors)
                        msi_neighbors = np.stack(msi_neighbors)
                        graph_meta = build_local_graph_metadata(center_coord, neighbor_coords, hop_ids, valid_neighbor_mask)

                        dataset.append((img_path, rna_temp, msi_temp, rna_neighbors, msi_neighbors, names[i] + '/' + key, graph_meta))

        return dataset

if __name__ == '__main__':
    dataset = SMA()
