from rdkit import rdBase, Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from matplotlib.colors import ColorConverter
import networkx as nx
import argparse
import multiprocessing
from rdkit import Chem
import numpy as np


def mol_to_nx(mol):
    G = nx.Graph()
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   atomic_num=atom.GetAtomicNum(),
                   formal_charge=atom.GetFormalCharge(),
                   chiral_tag=atom.GetChiralTag(),
                   hybridization=atom.GetHybridization(),
                   num_explicit_hs=atom.GetNumExplicitHs(),
                   is_aromatic=atom.GetIsAromatic())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType())
    return G, nx.adjacency_matrix(G).todense(), G.edges, G.nodes  # networks的图对象，邻接矩阵，边索引，节点索引


def nx_to_mol(G, smiles=True):
    """
    把network的Graph对象变成Mol对象或Smiles
    :param G: Graph对象
    :param smiles: 是否返回Smiles字符串，True为返回字符串，否则返回Mol对象
    :return:
    """
    mol = Chem.RWMol()
    atomic_nums = nx.get_node_attributes(G, 'atomic_num')
    chiral_tags = nx.get_node_attributes(G, 'chiral_tag')
    formal_charges = nx.get_node_attributes(G, 'formal_charge')
    node_is_aromatics = nx.get_node_attributes(G, 'is_aromatic')
    node_hybridizations = nx.get_node_attributes(G, 'hybridization')
    num_explicit_hss = nx.get_node_attributes(G, 'num_explicit_hs')
    node_to_idx = {}
    for node in G.nodes():
        a = Chem.Atom(atomic_nums[node])
        a.SetChiralTag(chiral_tags[node])
        a.SetFormalCharge(formal_charges[node])
        a.SetIsAromatic(node_is_aromatics[node])
        a.SetHybridization(node_hybridizations[node])
        a.SetNumExplicitHs(num_explicit_hss[node])
        idx = mol.AddAtom(a)
        node_to_idx[node] = idx

    bond_types = nx.get_edge_attributes(G, 'bond_type')
    for edge in G.edges():
        first, second = edge
        ifirst = node_to_idx[first]
        isecond = node_to_idx[second]
        bond_type = bond_types[first, second]
        mol.AddBond(ifirst, isecond, bond_type)

    Chem.SanitizeMol(mol)

    return Chem.MolToSmiles(mol) if smiles else mol


def CH_smiles_to_adj(mol, padding=16):
    """
    获取一个分子的邻接矩阵
    :param mol: Mol对象或SMILES表达式
    :param padding: 邻接矩阵填充到某个值
    :return:
    """
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    adj = Chem.GetAdjacencyMatrix(mol)
    if padding is not None:
        assert padding >= len(adj), '现有分子的邻接矩阵尺寸已经超过设置的最大尺寸，请检查设置！'
        add_number = padding - len(adj)
        adj = np.pad(adj, ((0, add_number), (0, add_number)))
    return adj


def CH_adj_to_smiles(adj, node_list=None, smiles=True, remove_padding=False, select_longest=True):
    """
    把邻接矩阵转化为smiles或Mol对象
    :param adj: 邻接矩阵。如果只需要单键，只需要全为0和1的普通邻接矩阵，如果需要其他键，邻接矩阵中的元素需要表示键的类型，1为单键，2为双键，3为三键
    :param node_list: 原子列表，里面是原子序数如分子C1CCC1应该是[6,6,6,6],如果只需要碳氢分子则不需要传入
    :param smiles: 是否返回smiles，True返回smiles，否则返回Mol对象
    :param remove_padding: 是否去掉padding，默认为False，用非常粗暴的方法去掉padding，很可能会去掉的太多，建议不要用True
    :param select_longest: 保留最长的smiles
    """

    def remove_padding_of_adj(adj):
        """
        去掉邻接矩阵的填0
        思路是如果有一行全是0，那说明从这之后的行都是填充的，只取前面的行和列就行
        :param adj: 邻接矩阵
        :return: 去掉padding的邻接矩阵
        """
        index = None
        for idx, i in enumerate(adj):
            if sum(i) == 0:
                index = idx
                break
        return adj[:index, :index]

    def select_longest_smiles(smi):
        return max(smi.split('.'), key=len, default='C')

    if select_longest:
        remove_padding = False

    if remove_padding:
        adj = remove_padding_of_adj(adj)

    mol = Chem.RWMol()

    # add atoms to mol and keep track of index
    if node_list is None:
        node_list = [6] * len(adj)  # 全设置为C
    node_to_idx = {}
    for i in range(len(node_list)):
        a = Chem.Atom(node_list[i])
        molIdx = mol.AddAtom(a)
        node_to_idx[i] = molIdx

    for ix, row in enumerate(adj):
        for iy, bond in enumerate(row):

            # only traverse half the matrix
            if iy <= ix:
                continue

            # add relevant bond type (there are many more of these)
            if bond == 1:
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], Chem.rdchem.BondType.SINGLE)
            elif bond == 2:
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], Chem.rdchem.BondType.DOUBLE)
            elif bond == 3:
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], Chem.rdchem.BondType.TRIPLE)

    fake_smiles = Chem.MolToSmiles(mol)  # 这个时候可能有荣誉的C，比如 C.C.CCC1C2CCC2C2C(C)C12
    if select_longest:
        fake_smiles = select_longest_smiles(fake_smiles)

    try:
        Chem.SanitizeMol(Chem.MolFromSmiles(fake_smiles))
    except:
        return 'error smiles'

    return fake_smiles if smiles else Chem.MolFromSmiles(fake_smiles)


def dense_adj_to_sparse_adj(adj):
    adj = adj - np.diag([1] * len(adj))
    adj = np.where(adj > 0.5, 1, 0)
    return adj


if __name__ == '__main__':

    print('rdkit version: ', rdBase.rdkitVersion)

    # # example
    # g, adj, edge_index, node_index = mol_to_nx('CC1C=CC1C')
    # print(adj)  # 邻接矩阵
    # print(g.nodes)  # 节点索引
    # print(g.nodes[0])  # 节点性质
    # print(g.edges)  # 边索引
    # print(list(g.edges))  #
    # print(g.get_edge_data(*list(g.edges)[0]))  # 解包
    # print(g.get_edge_data(0, 1))  # 直接给原子索引，效果跟解包是一样的
    # print(g.get_edge_data(0, 1, 'bond_type'))  # 指定取什么性质
    #
    # print(nx_to_mol(g))


    adj = CH_smiles_to_adj('CC1C=CCN1CC', padding=16)

    smi = CH_adj_to_smiles(adj=adj, node_list=[6, 6, 6, 6, 6, 7, 6, 6])

    print(adj)
    print(smi)