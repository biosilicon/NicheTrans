import os

import numpy as np
import pandas as pd
import scanpy as sc

import sklearn.neighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from scipy.sparse import issparse

from collections import defaultdict


def assign_spot_types(adata, n_pcs=30, n_neighbors=15, resolution=0.5, verbose=True):
    """
    Classify each spot into a discrete spot type via Leiden clustering on the
    transcriptomic PCA embedding.

    The clustering is performed on the pre-processed (normalised + log1p)
    expression matrix stored in ``adata.X``.  Only HVGs should have been
    selected *before* calling this function so that the PCA is meaningful.

    Parameters
    ----------
    adata : AnnData
        Must contain log-normalised counts in ``adata.X``.
    n_pcs : int
        Number of principal components to compute and use for neighbour graph.
    n_neighbors : int
        Number of neighbours for the kNN graph used by Leiden.
    resolution : float
        Leiden resolution; higher values produce more (finer) clusters.
    verbose : bool
        Whether to print cluster statistics.

    Returns
    -------
    spot_type_ids : np.ndarray, shape (n_obs,), dtype int
        Integer cluster label for every spot (0-indexed).
    n_types : int
        Total number of unique spot types found.
    """
    sc.pp.pca(adata, n_comps=n_pcs)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.leiden(adata, resolution=resolution, key_added='spot_type')

    # Map string cluster labels ("0", "1", ...) to integers.
    spot_type_ids = adata.obs['spot_type'].astype(int).values
    n_types = int(spot_type_ids.max()) + 1

    if verbose:
        counts = np.bincount(spot_type_ids)
        print(f'  Spot-type clustering: {n_types} types  '
              f'(sizes: {counts.tolist()})')

    return spot_type_ids, n_types


def align_clusters_across_slices(adata_hvg_list, type_ids_list, n_types_list,
                                  similarity_threshold=0.5, verbose=True):
    """
    Align per-slice Leiden cluster IDs into a unified global cell-type ID space.

    After independent Leiden clustering on each slice, cluster 0 on slice A may
    correspond to a completely different population than cluster 0 on slice B.
    This function resolves that ambiguity by:

    1. Computing per-cluster centroid expression vectors (mean over HVGs).
    2. Designating the first slice as the reference (local IDs 闂?global IDs).
    3. For every subsequent slice, matching its clusters to the reference using
       cosine similarity + the Hungarian algorithm.
    4. Assigning new global IDs to unmatched clusters.

    Parameters
    ----------
    adata_hvg_list : list[AnnData]
        HVG-filtered AnnData objects, one per slice.
    type_ids_list : list[np.ndarray]
        Per-spot integer Leiden labels for each slice.
    n_types_list : list[int]
        Number of clusters found in each slice.
    similarity_threshold : float
        Minimum cosine similarity to accept a Hungarian match.
    verbose : bool
        Print alignment diagnostics.

    Returns
    -------
    global_mapping : dict[(int, int), int]
        ``(slice_idx, local_cluster_id) -> global_cell_type_id``.
    n_global_types : int
        Total number of distinct global cell-type IDs.
    alignment_info : dict
        Diagnostic information (similarity matrices, matched pairs).
    """
    n_slices = len(adata_hvg_list)

    # 闂佸啿鍘滈崑鎾绘煃閸忓浜?Step 1: compute cluster centroids per slice 闂佸啿鍘滈崑鎾绘煃閸忓浜鹃梺鍐插帨閸嬫捇鏌嶉崗澶婁壕闂佸啿鍘滈崑鎾绘煃閸忓浜鹃梺鍐插帨閸嬫捇鏌嶉崗澶婁壕闂佸啿鍘滈崑鎾绘煃閸忓浜鹃梺鍐插帨閸嬫捇鏌嶉崗澶婁壕闂佸啿鍘滈崑鎾绘煃閸忓浜鹃梺鍐插帨閸嬫捇鏌嶉崗澶婁壕闂佸啿鍘滈崑鎾绘煃閸忓浜鹃梺鍐插帨閸嬫捇鏌嶉崗澶婁壕闂佸啿鍘滈崑?
    centroids_list = []  # centroids_list[s] shape: (n_types_list[s], n_hvg)
    for s in range(n_slices):
        X = adata_hvg_list[s].X
        if issparse(X):
            X = X.toarray()
        n_clusters = n_types_list[s]
        ids = type_ids_list[s]
        centroids = np.zeros((n_clusters, X.shape[1]))
        for c in range(n_clusters):
            mask = (ids == c)
            if mask.sum() > 0:
                centroids[c] = X[mask].mean(axis=0)
        centroids_list.append(centroids)

    # 闂佸啿鍘滈崑鎾绘煃閸忓浜?Step 2: reference slice (slice 0) keeps its local IDs 闂佸啿鍘滈崑鎾绘煃閸忓浜鹃梺鍐插帨閸嬫捇鏌嶉崗澶婁壕闂佸啿鍘滈崑鎾绘煃閸忓浜鹃梺鍐插帨閸嬫捇鏌嶉崗澶婁壕闂佸啿鍘滈崑鎾绘煃閸忓浜鹃梺鍐插帨閸?
    global_mapping = {}
    for c in range(n_types_list[0]):
        global_mapping[(0, c)] = c
    next_global_id = n_types_list[0]

    alignment_info = {'reference_slice': 0, 'matches': {}}

    # 闂佸啿鍘滈崑鎾绘煃閸忓浜?Step 3: align each subsequent slice to the reference 闂佸啿鍘滈崑鎾绘煃閸忓浜鹃梺鍐插帨閸嬫捇鏌嶉崗澶婁壕闂佸啿鍘滈崑鎾绘煃閸忓浜鹃梺鍐插帨閸嬫捇鏌嶉崗澶婁壕闂佸啿鍘滈崑鎾绘煃閸忓浜鹃梺鍐插帨閸嬫捇鏌嶉崗澶婁壕
    ref_centroids = centroids_list[0]

    for s in range(1, n_slices):
        cur_centroids = centroids_list[s]
        # Cosine similarity: (n_cur, n_ref)
        sim_matrix = cosine_similarity(cur_centroids, ref_centroids)
        cost_matrix = 1.0 - sim_matrix

        # Hungarian algorithm for optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        slice_matches = []
        matched_rows = set()
        for r, c_ref in zip(row_ind, col_ind):
            sim_val = sim_matrix[r, c_ref]
            if sim_val >= similarity_threshold:
                # Map this slice's cluster r to the same global ID as ref cluster c_ref
                global_mapping[(s, int(r))] = global_mapping[(0, int(c_ref))]
                matched_rows.add(r)
                slice_matches.append((int(r), int(c_ref), float(sim_val)))

        # Unmatched clusters in this slice get new global IDs
        for c in range(n_types_list[s]):
            if c not in matched_rows:
                global_mapping[(s, c)] = next_global_id
                next_global_id += 1
                if verbose:
                    print(f'  Slice {s}, local cluster {c}: no match '
                          f'(new global ID {global_mapping[(s, c)]})')

        alignment_info['matches'][s] = slice_matches

        if verbose:
            print(f'  Slice {s} alignment: {len(slice_matches)}/{n_types_list[s]} '
                  f'clusters matched to reference')
            for r, c_ref, sim_val in slice_matches:
                print(f'    local {r} -> ref {c_ref} (global {global_mapping[(0, c_ref)]}), '
                      f'cosine sim = {sim_val:.4f}')

    n_global_types = next_global_id
    if verbose:
        print(f'  Cross-slice alignment complete: {n_global_types} global cell types')

    return global_mapping, n_global_types, alignment_info


# return the neighborhood nodes
def Cal_Spatial_Net_row_col(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True):
    """
    闂佸搫绉烽～澶婄暤?spot 闂?array_row 闂?array_col 闂佺鍕闁绘牭缍佸鎼佸礋椤愩倗澶勭紓浣告湰濡炶棄螞閸ф鐒界紓浣姑径宥夋煕閵夈倕瀚庨柍?

    闂佸憡鐟ラ崐褰掑汲?
        adata: AnnData 闁诲海鏁搁、濠囨寘閸曨垱鏅€光偓閸愮偓鐭楀┑?adata.obs 婵炴垶鎼╅崢鑲┾偓鍨耿瀹?`array_row` 闂?`array_col`闂?
        rad_cutoff: 閻?model='Radius' 闂佸搫鍟﹢瑙勭箾閸ヮ剚鍋ㄩ柕濞у嫮鏆犻梺鍛婎殔閿曘倗娆㈤悙鐑樷挀闁割偅绺鹃崑鎾剁箔鐞涒€充壕?
        k_cutoff: 閻?model='KNN' 闂佸搫鍟﹢瑙勭箾閸ヮ剚鍋ㄩ柕濞у嫮鏆犻柡澶嗘櫆閸ㄥ潡宕戦敂鐣屸枖妞ゅ繐鎳忓▓鍫曟煏?
        model: 闂備緡鍘剧划顖滄暜閹绢喖鐐婇柣鎰靛墰閳ь剦鍨伴娆撴倻濡崵鍘甸悗娈垮枛妤犲繒妲愬┑瀣哗妞ゆ牗绋戦惁?'Radius' 闂?'KNN'闂?
        verbose: 闂佸搫瀚烽崹浼村箚娓氣偓楠炲秹骞橀崘鑼偑閻庣偣鍊曢幖顐⒚瑰Ο鑽も攳闁斥晛鍟╃槐鏍煏?

    闁哄鏅滈弻銊ッ?
        temp_dic: 婵?"row_col" 婵炴垶鎸诲浠嬪极椤撱垺鍎嶉柛鏇ㄥ灠娴犳悂鏌熼幁鎺戝姎闁烩姍鍥х闂傚牃鏅濈粈澶愭煕婵犲洦锛熼悹鎰枛閹嫰骞栨担闀愭 spot 闂?"row_col" 闂佸憡甯楅〃澶愬Υ閸愵喖违?
    """

    assert(model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')

    # 闂佸憡鐟﹂悧鏇㈠吹椤撶倫鎺曠疀鎼淬劌娈?spot 闂佹眹鍔岀€氼亞鑺遍埡鍐＜閻庣數纭堕弫鍕⒒閸屾稒缍戞繛鍏碱殜瀵粙宕舵搴ｎ槷濡ょ姷鍋犻崺鏍汲閿濆鍋犻柛鈩兠悘鍥煕濮橆剚鎹ｉ柣銏㈢帛濞煎骞嬮敃鈧禒鎼佹煙閸忚偐鐭岄柛灞诲姂楠炲秹鍩€椤掑嫭顥嗛柍褜鍓熼幆?DataFrame闂?
    coor = pd.DataFrame(np.stack([adata.obs['array_row'], adata.obs['array_col']], axis=1))

    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    if model == 'Radius':
        # 闂佺硶鏅炲銊ц姳椤掑嫬纭€濠电姴鍊荤粣鐐烘煙閸忚偐鐭岄柛灞诲姂閺屽洨鎷犻懠顒佺嫍闂佹寧绋戦惌浣烘崲閹达箑鐐婇柣鎰嚟濡层劌鈽夐幙鍐х敖闁稿缍佸畷鐑藉Ω瑜忛懜鍫曟倵鐟欏嫯澹樼€规洘鍔曢銉╁礋椤愩垺鏆ラ梺姹囧妼鐎氼厽鏅跺澶婂珘濠㈣埖鍔曟禒鎼佹倶閻愬弶鍣圭€殿噣顥撻幑鍕攽閸偆鈧煡鏌?
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):      #indices闂佸搫瀚烽崹鍫曞磻閿斿彞娌柛娑卞灠閸嬪秶鈧鍠楀ú鏍垂椤忓棙鍋?
            # 濠殿噯绲界换瀣博鐎靛憡鍋樼€光偓閸愭儳鏁归悷? 閻熸粎澧楅幐鍛婃櫠閻樼粯鍊烽柛锔诲幖閸嬪秶鈧鍠楀ú蹇涘焵椤戣法绐旈柛瀣跺娴狅箓宕ㄩ姘寲缂備椒绌堕崹鍦閳哄懎违濞达絿鏌夐埛鍫ユ煠閺夋寧鍤€缂佹劖绋撶划瀣ч崶锝呬壕?
            KNN_list.append(pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])))
    
    if model == 'KNN':
        # KNN 婵炴潙鍚嬬喊宥嗕繆閹间焦鍊风痪顓炴噹濞堜即妫呴澶婁簵缂佲€冲暟缁螖閸曨厾顔掗梺鍝勭墐閸嬫捇寮堕埡鍌氱伌闁稿绠撻弫宥囦沪閼恒儳顦┑顔界缚閸婃稓鎹㈤弽顓熺厒閻忕偟鍘ч埢蹇涙煟?k_cutoff + 1闂?
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff+1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices.shape[1],indices[it,:], distances[it,:])))

    # 闂佸憡鑹鹃悧鍡涙嚐閻旂厧绠ラ柍褜鍓熷鍨緞鐏炴垝鍖栭梺姹囧妼鐎氫即宕戦敃鍌氱闁靛／鍛皾闂佸搫顑嗙划锝囨濠靛宓侀柛鎾茶兌閸╃姴鈽夐幘顖氫壕闂佸憡甯楅〃鍛村箖閺囥垹违?
    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']     #Cell1闂佸搫瀚烽崹顖炴嚈閹达絿鐤€闁告劦浜為惌搴ㄦ煠瀹曞洤鍓崇紒杈ㄦ€€ell2闂佸搫瀚烽崹鍫曞磻閿斿彞娌柛娑卞灣閻酣鏌ゅ畷鍥у壋缂?Distance闂佸搫瀚烽崹鐢电玻濞戞氨鐭?

    Spatial_Net = KNN_df.copy()
    # 闂佸憡顭囩划顖滄暜閳ь剟鏌ゆ總澶夌盎缂佽绶氬畷姘跺箯鐏炶姤顔囬悗瑙勭摃娴滎剙鈻撻幋鐐翠氦闁绘劕澧庨悵鍫曟煥濞戞ê顨欑紒顔煎缁屽崬鈹戦崼鐔哥暚闂佹椿浜為崰搴ㄦ偪閸曨垱鐒介悹鍥皺濠€鐑芥煕韫囨挾锛嶉梺顔芥尦婵?
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance']>0,]
    # 闁诲繐绻愬Λ娆撳汲閿濆鏋侀柟娈垮枛閸嬪秶鈧鍠楀ú妯何ｈ娴滄悂宕熼锝囶槬 adata.obs 婵炴垶鎼╅崢鑲╂暜椤愶絾鍙忛悗锝庡墯閻?spot 闂佸憡鑹剧粔鎯扳叿闂?
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' %(Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' %(Spatial_Net.shape[0]/adata.n_obs))

    # 闁诲繐绻愬Λ婊呯博閻旂儤鍋橀柕濞垮€楃粻浠嬫倵濞戞顏堝春?AnnData 婵炴垶鎼╅崣蹇曟濠靛洨鐟瑰璺烘憸閼归箖鏌涘顒佹崳闁汇垻绮鍕吋閸ャ劍娈㈤梺?
    adata.uns['Spatial_Net'] = Spatial_Net

    # 闂佸搫顑呯€氼剛绱?"row_col" -> 闂備緡鍘剧划顖滄暜?"row_col" 闂佸憡甯楅〃澶愬Υ閸愵喗鍎嶉柛鏇ㄥ亞閹界喖鏌涜箛鎾村暈闁绘顭堥锝堢疀濮樿京鐐曢梺鍛婂灦娴滀粙鍩€?
    temp_dic = defaultdict(list)
    for i in range(Spatial_Net.shape[0]):

        center = Spatial_Net.iloc[i, 0]
        side = Spatial_Net.iloc[i, 1]

        # 闂?array_row 闂?array_col 闂佺懓鍢查崥瀣暜閹绢喖绀?spot 闂佹眹鍔岀€氼亞鑺遍埡鍐＜闁哄鍨剁紞蹇涙煛瀹ュ懏鎼愰柟顔芥礈缁棃鎳滈钘変壕?
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

            # 闁诲簼绲婚～澶愭儊閳ユ剚鍤曢柣妯诲絻閻庡ジ鏌ｅΔ鈧ú銈呯暦閻斿吋鍋戞い鎺嗗亾濞?HVG 缂備焦绋掗惄顖炲焵椤掆偓椤︻垶骞忔导鏉戝唨闁搞儜鍐╃彲 scRNA 婵☆偅婢樼€氼剟藝閳哄懏鍋犻柛鈩冨姀閸?
            # 闂佸憡鑹鹃柊锝咁焽閻楀牆顕辨慨姗嗗墻閸?3 閻庢鍠氭慨鎾垂韫囨稒鍋嬮柛銉㈡櫆閻?HVG 缂傚倷鐒﹂幐濠氭倶婢舵劕鐭楅柡宥忛檮閸炲姊婚崱妤侊紨缂佽鲸绻堝畷锝夘敂閸愵亞顔旈梺浼欑稻閻熴劌鈻旈弴銏犵闁搞儮鏅涢。濠氭⒑椤旂偓娅曢懣娆撴倵鐟欏嫭鐨戞繛鍫熷灴瀹曟椽宕崟顒傤槶闂?
            sc.pp.highly_variable_genes(adata_rna, flavor="seurat_v3", n_top_genes=n_top_genes)
            sc.pp.normalize_total(adata_rna, target_sum=1e4)
            sc.pp.log1p(adata_rna)

            rna_adata_list.append(adata_rna)
            rna_highly_variable_list.append(adata_rna.var['highly_variable'].values)

            adata_msi = sc.read_h5ad(os.path.join(msi_path, 'metabolite_' + slide + '.h5ad'))
            adata_msi.var_names_make_unique()
            # MSI 婵炴垶姊婚崰鎰偓鍨矒瀹曟岸宕堕埡渚囨毈闂佺粯鐟崜娑㈡偟濞戞氨椹虫繛鎴欏灮瑜邦垶鏌涘▎鎯峰ジ宕鍌涘闁靛繈鍨归埛鏍煥濞戞瀚伴柟顔筋殜濡啴濮€閻樺啿鈧亶鏌″鍛Щ鐟滄澘娲︾粋宥夊Χ婢跺鍋侀梺?
            # 闁哄鏅滈悷鈺呭闯閻戣棄鐭楁い蹇撳缁?log1p闂佹寧绋戞總鏃傜箔婢舵劕绀冪€广儱鎳嶇划?total-count normalize闂?
            sc.pp.highly_variable_genes(adata_msi, flavor="seurat_v3", n_top_genes=n_top_targets)
            sc.pp.log1p(adata_msi)

            msi_adata_list.append(adata_msi)
            msi_highly_variable_list.append(adata_msi.var['highly_variable'].values)


        ##############
        # 婵☆偅婢樼€氼剟宕㈠☉姘辩＜闁绘梹妞块崥鈧梺鍦暯閸嬫捇鏌￠崼婵愭Ц闁搞劌绻橀幃?RNA 闂佹眹鍔岀€氼剟宕ｈ箛鏇氭勃闁逞屽墴瀹曟悂宕惰閸?闂佸搫鍊婚幊鎾澄熸繝鍥ㄦ櫖閻忕偠鍋愰埞鎺懨瑰鍐€楅柟顔筋殘缁辨帡顢橀悙鎵┏闂佺硶鏅涢幖顐⒚哄鍛／鐟滃酣鎮ラ敐澶婄闁糕剝顨呴褔鏌?
        # 閻熸粎澧楅幐鍛婃櫠閻樿櫕浜ゆ繛鍡楅叄閸ゅ绱掗銏╃吋闁革絿鍏橀悰顕€鎳滈悽鍨挄闂佸搫鐗嗛ˇ顖炴偪閸曨垱鈷旈柛娑卞灡閺嗗繘鏌涢幒鎾愁€滅紒缁樼墬缁嬪濡堕崟顖氭疂闂備焦褰冩蹇曟濠靛牅娌柣鎰閼规儳菐閸ャ劎绠橀柡鍡忓亾闂佹眹鍔岀€氼喚鍒掗搹瑙勫闁炽儴寮撶换鍡涙煙椤撗冪仜闁?
        temp = np.concatenate([ rna_adata_list[0].X.toarray(), rna_adata_list[1].X.toarray(), rna_adata_list[2].X.toarray()], axis=0)
        self.rna_mean, self.rna_std = temp.mean(axis=0)[None, ], temp.std(axis=0)[None, ]
        
        temp_mask = (self.rna_std == 0)
        # 闂備緡鍓欓悘婵嬪储閵堝瑙﹂幖杈剧悼閺侀箖鏌ゅЧ鍥у姎濞?z-score 闂佸搫绉村ú銈夊闯椤栫偛绀岄柡宥冨妽椤ρ囨煕閹达絾顏犳鐐叉喘濮婁粙濡堕崟顏嗛瀺 0闂?
        self.rna_std[temp_mask] = 1

        ###############

        # 闂佸憡鐟禍娆戞崲濮樿埖鍋?3 閻庢鍠氭慨鎾垂韫囨稒鍋嬮柛銉戝啫缍戦梺鍛婅壘閻厧鈻撻幋鐑嗘畻婵☆垰鎼紞渚€鏌涢埡浣规儓婵?婵°倕鍊归敋鐟滄澘顦扮粋鎺楁晲閸儲鈻奸梺缁樸仜閺咁亞妲?
        # 闁哄鏅滈悷锕傛偋鏉堚晜濯兼い鎾跺Х閻﹪鏌涘鍐╂拱缂佷礁顕幏鐘诲即閳╁啫鍔欓梺闈╄礋閸斿矂骞冮幘瀵糕枖闁逞屽墯缁鸿棄螖閸忥附姊婚埀顒冾潐濮樸劍绔熼幒鎴殫濞达綀銆€閺佸嫰姊婚崒娑欏唉闁革絿鍏樻俊?
        self.rna_mask =  rna_highly_variable_list[0] & rna_highly_variable_list[1] & rna_highly_variable_list[2]
        self.msi_mask = msi_highly_variable_list[0] & msi_highly_variable_list[1] & msi_highly_variable_list[2]

        # 闂佸啿鍘滈崑鎾绘煃閸忓浜?Spot-type classification + cross-slice alignment 闂佸啿鍘滈崑鎾绘煃閸忓浜鹃梺鍐插帨閸嬫捇鏌嶉崗澶婁壕闂佸啿鍘滈崑鎾绘煃閸忓浜鹃梺鍐插帨閸嬫捇鏌嶉崗澶婁壕闂佸啿鍘滈崑鎾绘煃閸忓浜鹃梺鍐插帨閸嬫捇鏌嶉崗澶婁壕闂佸啿鍘滈崑鎾绘煃閸忓浜鹃梺鍐插帨閸嬫捇鏌嶉崗澶婁壕
        # Run Leiden clustering on each slice independently (using only HVGs),
        # then align cluster IDs across slices so that biologically similar
        # clusters share the same global cell-type ID.

        adata_hvg_list = []
        type_ids_list = []
        n_types_per_slide = []

        for idx, slide in enumerate(training_slides + testing_slides):
            adata_rna = rna_adata_list[idx]
            # Work on HVG-filtered copy so PCA is computed on the same genes
            # that will be used for training.
            adata_hvg = adata_rna[:, self.rna_mask].copy()
            type_ids, n_t = assign_spot_types(adata_hvg, verbose=True)
            adata_hvg_list.append(adata_hvg)
            type_ids_list.append(type_ids)
            n_types_per_slide.append(n_t)

        # Align local cluster IDs into a unified global space.
        global_mapping, n_global_types, alignment_info = align_clusters_across_slices(
            adata_hvg_list, type_ids_list, n_types_per_slide, verbose=True)
        self.n_spot_types = n_global_types
        self.global_mapping = global_mapping  # (slice_idx, local_id) -> global_id

        # Build per-spot type ID dict using globally aligned IDs.
        all_spot_type_ids = {}   # key: "slide/row_col" -> int global_cell_type_id
        for idx, slide in enumerate(training_slides + testing_slides):
            adata_rna = rna_adata_list[idx]
            array_row = adata_rna.obs['array_row'].values
            array_col = adata_rna.obs['array_col'].values
            for i in range(adata_rna.shape[0]):
                coord_key = str(int(array_row[i])) + '_' + str(int(array_col[i]))
                local_id = int(type_ids_list[idx][i])
                all_spot_type_ids[slide + '/' + coord_key] = global_mapping[(idx, local_id)]

        print(f'  Total global cell types used for embedding: {self.n_spot_types}')

        self.training = self._process_data(
            rna_adata_list[0:2], msi_adata_list[0:2],
            training_slides, all_spot_type_ids)
        self.testing = self._process_data(
            rna_adata_list[2:], msi_adata_list[2:],
            testing_slides, all_spot_type_ids)

        self.rna_length = (self.rna_mask * 1).sum()
        self.msi_length = (self.msi_mask * 1).sum()
        # Aliases expected by the training script.
        self.source_length = int(self.rna_length)
        self.target_length = int(self.msi_length)
        # source_panel/target_panel 婵烇絽娲︾换鍌炴偤閵娿儙鐔煎灳瀹曞洠鍋撻悜鑺ュ剳闁绘棃顥撻弶钘壝归敐鍫熺《闁轰降鍊濋幆鍐礋椤斿墽鐐曢梺绋跨箞閸庨亶鎮㈤埀顒勬煕閵壯冃㈤柟顔芥礋瀹曨亜鐣濋崘顏嗙倳闂佸憡鍨煎銊╁船椤掑倹瀚柕蹇嬪灩閳锋牠鏌涘顒傗枌缂?
        # 闁荤姳绶￠崢鍓у垝瀹€鍕Е閹艰揪绲鹃悾閬嶆煕濞嗘ê鐏ユい顐㈩儔瀹曠娀寮借鐎?attribution 闂佸憡甯掑Λ娆撴倵娴犲鐒鹃柟瀛樺笧缁愭銆掑顓犵畾缂佸倸妫欏璇测枎鎼淬倐鎷℃繛鎴炴惄娴滅偤宕掗妸銉殨闁哄洦菤閸?
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
            # 闂?"row_col" 婵炶揪绲剧划鍫㈡嫻閻旇櫣纾奸柣鏂垮椤忛亶鏌涜閸旀牠鎮ラ敐澶嬬叆妞ゆ洖妫涚粈澶娿€掑顑句粧缂?RNA闂侀潧妫旀惔娣狪 闂佸憡绮岄惌鍌炲磻閿曞倸绠抽柕澶堝劜缁傚牓鏌熺粙娆炬Ш闁宠鐗犲鑽ょ礄閻樼數孝缂傚倸鍠氶崰鏍敋椤旂瓔鐎查柟鎵虫杹閸?
            dictionary[str(int(array_row[i])) + '_' +  str(int(array_col[i])) ] = array[i]

        return dictionary

        
    def _process_data(self, rna_adata_list, msi_adata_list, names, all_spot_type_ids):
        """Build the list of per-spot sample tuples.

        Each element is::

            (img_path, rna_temp, msi_temp, rna_neighbors, msi_neighbors,
             spot_type_id, sample_id)

        ``spot_type_id`` is an int in ``[0, n_spot_types)``.
        """

        dataset = []

        for i in range(len(rna_adata_list)):

            rna_temp_adata = rna_adata_list[i]
            msi_temp_adata = msi_adata_list[i]

            rna_dic = self._dictionary_data(rna_temp_adata, rna=True)
            msi_dic = self._dictionary_data(msi_temp_adata)

            # graph = Cal_Spatial_Net_spatial_cite_seq(rna_temp_adata,  k_cutoff=8, model='KNN')
            graph_1 = Cal_Spatial_Net_row_col(rna_temp_adata,  rad_cutoff=2**(1/2), model='Radius')    #婵炴垶鎹佸銊у垝閸喓鈻曢柛顐ゅ枑绗戦梺鍝勭Ф閹虫捁銇?闂佹寧绋撻崰鏇犳崲閺嶎厼鍐€闁炬艾鍊婚悷婵嬫煛閸曢潧鐏熺紒鍙樺嵆閺屽懏寰勬繝鍌滃帓婵炴垶鎸撮崑鎾绘煕?婵炴垶鎼╂禍鐐靛垝韫囨稒鍤勯柣鎰级閸嬪鏌?
            graph_2 = Cal_Spatial_Net_row_col(rna_temp_adata,  rad_cutoff=2, model='Radius')

            rna_keys, msi_keys = rna_dic.keys(), msi_dic.keys()

            for key in rna_keys:
                # 闂佸憡鐟禍娆戞崲濮樿埖鍋?RNA 婵?MSI 闂備緡鍠涘Λ鍕偤閵娾晛鎹堕柕濞у嫮鏆犻梺绉嗗嫬濮堥柣鏍电秮閺佸秶浠﹂懖鈺冾啍闁荤姴娲ｉ懗璺好洪崘顔藉創闁挎繂瀚换鍡涙煕濞嗘挻鏁辨俊鎻掓憸缁艾煤椤忓拑绱甸柣搴ㄦ涧缂嶅﹦绱為崼銉﹀剭闁告洍鏂侀崑?
                if key not in msi_keys:
                    continue
                else:
                    # 婵炴垶鎼╅崢鑲╃紦?spot 婵炲濮撮幊宥囨崲濮樿埖鍋╂繛鍡楃箰瑜版瑩鏌涘顒傜劯闁绘繍鍣ｅ畷锝呂熼悡搴閻庣敻顣︾欢銈囨濠靛洦濯存繝濞惧亾閻犳劗鍠庤灒闁炽儱纾埀顒傚厴閹啴宕熼鍓х倳闂?闂佺儵鏅╅崰妤呮偉閿濆违?
                    rna_temp = rna_dic[key][self.rna_mask]
                    msi_temp = msi_dic[key][self.msi_mask]

                    # quality check for rna and msi
                    # 闂佸吋鐪归崕鎶芥偤椤愩倐鍋撻悷鎷屽闁诡喗顨堥幏?spot 闂侀潻璐熼崝瀣箲閵忥紕鈻旈柍褜鍓欒灒闁斥晛鍟犻崑鎾存媴妞嬪海鎲梺?0闂佹寧绋戦懟顖炲垂椤栨粍濯奸柕鍫濆缁€瀣箾鐏炵澧叉繝鈧笟鈧鍨緞鐎ｎ偅鐝℃繛锝呮礌閸撴繃瀵奸崨鏉懳?
                    if rna_temp.sum() == 0 or msi_temp.sum() == 0:
                        continue
                    else:
                        # 濠殿噯绲界换瀣煂濠婂懐鐭氭繛宸簼閿涚喎霉閿濆懐肖闁汇倕妫欏璇参熼崷顓犵崶闂佺绻愰悿鍥ㄧ閸喓鈻旈柍褜鍓欓?histology patch闂佹寧绋戦懟顖炴嚋娴兼潙绠ｉ柟閭﹀墯缁傚牓鏌?+ 缂傚倷绀佺€氼剟顢楅悢鍏煎殏闁哄倹瀵ч崐銈夊级閸喐灏柛娆忔婵?
                        img_path = os.path.join(self.path_img, names[i], key + '.png')

                        # Retrieve the globally aligned cell-type ID for this spot.
                        # Fall back to type 0 only when the coordinate is absent
                        # (for example after downstream QC filtering).
                        global_key = names[i] + '/' + key
                        spot_type_id = all_spot_type_ids.get(global_key, 0)

                        rna_neighbors, msi_neighbors = [], []

                        neighbors_1, neighbors_2 = graph_1[key], graph_2[key]
                        # 缂備焦顨忛崗娑氳姳閳哄懎鎹堕柛顐犲劚娴犳悂鎮橀悙鍙夊櫣妤犵偞鎹囬獮鎺撳緞婵犲倸娈ラ柣鐐村嚬閸嬪棝顢栭崶銊р枖闁逞屽墴瀹曠兘宕奸敐搴㈣埞闂佺儵鏅滈悧婊冣枔閹达附鍊烽柣褍鎽滅粈澶愭⒑椤掆偓閻忔繈宕㈤妶澶嬬厒鐎广儱鎷嬪Σ濠氭煏?
                        neighbors_2 = [item for item in neighbors_2 if item not in neighbors_1]

                        # connect to the first round
                        for j in neighbors_1:
                            # 闂備緡鍙€椤曆囨儑瀹曞洨纾介柛婵嗗娴滃ジ鏌￠崘鈺佸姸閽樼喖姊婚崱姘【闁诡喗绮撻弻宀冪疀閵壯咁槷婵烇絽娲︾换鍐偓鍨瀹曞爼宕崟顓熸瘞闁哄鐗婇幐鎼佸矗閸℃瑧纾奸柡澶嬪灥椤斿﹪鏌?
                            if j not in rna_keys:
                                rna_neighbors.append(np.zeros_like(rna_temp))
                            else:
                                rna_neighbors.append(rna_dic[j][self.rna_mask])

                            if j not in msi_keys:
                                msi_neighbors.append(np.zeros_like(msi_temp))
                            else:
                                msi_neighbors.append(msi_dic[j][self.msi_mask])

                        # 缂備焦顨忛崗娑氱博閹绢喖鎹堕柛顐ｇ箖缁佸ジ鎮楃憴鍕叝缂佺粯宀搁幃?4 婵炴垶鎼╂禍婵婃綍婵炶揪绲界粙鍕濠靛洨鈻旂€广儱鐗嗛崰鏇㈡煕閹烘挾鎳嗛挊鐔兼⒒閸℃氨鍫柍?
                        if len(neighbors_1) != 4:
                            for _ in range(4-len(neighbors_1)):
                                rna_neighbors.append(np.zeros_like(rna_temp))
                                msi_neighbors.append(np.zeros_like(msi_temp))

                        # connect to the second round
                        for j in neighbors_2:
                            # 缂備焦顨忛崗娑氳姳閳哄懎鎹堕柛顐ｇ矌閻熴垻绱掑Δ濠傚幐缂佹梹鎸冲畷鐑藉醇閵忥紕銈查梺鍛婅壘閻妲愬┑鍥┾枙闁绘棁娉涢埢蹇涙煟椤剙濡兼繛瀛樺姉閳ь剝顫夊畝鎼佸汲鏉堛劍鍎熼柨鏃傚亾閻ｉ亶姊洪鈽嗗殭闁绘挻鐟х槐鎾诲冀椤愩倕鐏ｉ梺?
                            if j not in rna_keys:
                                rna_neighbors.append(np.zeros_like(rna_temp))
                            else:
                                rna_neighbors.append(rna_dic[j][self.rna_mask])

                            if j not in msi_keys:
                                msi_neighbors.append(np.zeros_like(msi_temp))
                            else:
                                msi_neighbors.append(msi_dic[j][self.msi_mask])

                        # 缂備焦顨忛崗娑氳姳閳哄懎鎹堕柛顐ｇ箖閸婇亶鏌″鍛Щ婵炲瓨鍔楅埀顒冾潐閻喚鎷?4 婵炴垶鎼╂禍婵婃綍婵炶揪绲界粙鍕濠靛鐐婇柣妯诲墯閸斿啴鏌熼鈧…鐑藉磻閿曞倸鏄ラ柣鏃€鐏氭禍锝夋倶韫囨挾绠叉俊?8闂?
                        if len(neighbors_2) != 4:
                            for _ in range(4-len(neighbors_2)):
                                rna_neighbors.append(np.zeros_like(rna_temp))
                                msi_neighbors.append(np.zeros_like(msi_temp))

                        # 闂佸搫鐗冮崑鎾剁磽娴ｅ摜澧旂紒鏂跨摠缁嬪顢旈崨顖氼棊闂佸搫鐗滈崜娑㈠极闁秵鏅?
                        # 婵炴垶鎼╅崢鑲╃紦妤ｅ啫鐐婇柟顖嗗啫澹?+ 婵炴垶鎼╅崢鑲╃紦?RNA + 婵炴垶鎼╅崢鑲╃紦?MSI + 8 婵?RNA 闂備緡鍙€椤曆囨儑?+ 8 婵?MSI 闂備緡鍙€椤曆囨儑?
                        # + spot_type_id + sample id 缂傚倷绀佺€氼參宕瑰璺何?
                        rna_neighbors = np.stack(rna_neighbors)
                        msi_neighbors = np.stack(msi_neighbors)

                        dataset.append((img_path, rna_temp, msi_temp, rna_neighbors, msi_neighbors,
                                        spot_type_id, names[i] + '/' + key))

        return dataset

if __name__ == '__main__':
    dataset = SMA()
