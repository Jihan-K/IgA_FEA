# -*- coding: utf-8 -*-
"""

@author: wnchang@iu.edu

- Outputs: flux, balance, loss value and loss plot in res_dir.
- Default model:
    grr    : data/pyrimidine_grr_path.csv  (pathway,reaction,gene; pathway = 'pyrimidine')
    matrix : data/pyrimidine_matrix.csv    (43 reactions)
    input  : input/IgA_all.csv
- Reactions in the matrix must all appear in the grr (checked at load time).
"""

# system lib
import argparse
import time
import warnings
import os

# tools
import torch
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functools import reduce
from tqdm import tqdm

# scFEA lib
from ClassFlux import FLUX # Flux class network
from util import pearsonr
from DatasetFlux import MyDataset



# hyper parameters
LEARN_RATE = 0.008
#EPOCH = 100
LAMB_BA = 1
LAMB_NG = 1
LAMB_CELL =  0
LAMB_RXN = 1

# random seed for network initialization and expression noise
SEED = 62424

#tissue specificity
tissueSpecific = [[0.2, 2], [0.1, 4]]

total_and = True
each_and = False


# The and rule reflects that some reactions need subunits
# It selects the least value among genes encoding subunits.
def andRule(mat, rules):
    mat['N/A'] = 0

    if total_and:
        for i in range(mat.shape[0]):
            tmp = []
            tmptmp = []
            for _ in rules:
                tmptmp += _

            if tmptmp != ['']:
                tmptmp = list(set(tmptmp))
                
                andMean = ((mat.iloc[[i], :].loc[:, tmptmp].mean()).iloc[0])/len(tmptmp)
                mat.iloc[[i], :].loc[:, tmptmp] = andMean
                
                mat.loc[mat.index[i], [g for g in mat.columns if g not in tmptmp]] = 0
            else:
                mat.loc[mat.index[i], [g for g in mat.columns if g not in tmp]] = 0

    if each_and:
        for i in range(mat.shape[0]):
            tmp = []
            for _ in rules:
                if len(_) == 1:
                    tmp += _
                else:
                    andMean = ((mat.iloc[[i], :].loc[:, _].mean()).iloc[0])/len(_)
                    mat.iloc[[i], :].loc[:, _] = andMean
                    mat.loc[mat.index[i], [g for g in mat.columns if g not in _]] = 0
                    tmp += _
            tmp = list(set(tmp))
            mat.loc[mat.index[i], [g for g in mat.columns if g not in tmp]] = 0

    del mat['N/A']

    return mat


def myLoss(r, c, lamb1 = 0.2, lamb2= 0.2, lamb3 = 0.2, lamb4 = 0.2, geneScale = None, reactionScale = None, dirConstraints = None, balConstraints = None):
    # balance constrain
    total1 = torch.pow(c, 2)
    total1 = torch.sum(total1, dim = 1) * balConstraints
    
    # non-negative constrain
    # Forward only reactions yield positive when estimated to be reverse, and vice versa.
    # Bidirectional reactions do not contribute to loss.
    error = torch.abs(r*dirConstraints) - (r*dirConstraints)
    total2 = torch.sum(error, dim=1) 
    
    # sample-wise variation constrain 
    diff = torch.pow(torch.sum(r, dim=1) - geneScale, 2)
    if sum(diff > 0) == r.shape[0]: # solve Nan after several iteraions
        total3 = torch.pow(diff, 0.5)
    else:
        total3 = diff
    
    # reaction-wise variation constrain
    if lamb4 > 0 :
        corr = torch.FloatTensor(np.ones(r.shape[0]))
        for i in range(r.shape[0]):
            # if pearson r cannot be calculated
            # (when no gene associated with any reaction is expressed)
            # exclude from loss function.
            if reactionScale[i, :].sum() == 0:
                corr[i] = 1
            else:
                corr[i] = pearsonr(r[i], reactionScale[i])
        corr = torch.abs(corr)
        penal_r_var = torch.FloatTensor(np.ones(r.shape[0])) - corr
        total4 = penal_r_var
    else:
        total4 = torch.FloatTensor(np.zeros(r.shape[0]))
    
    # loss
    loss1 = torch.sum(lamb1 * total1)
    loss2 = torch.sum(lamb2 * total2)
    loss3 = torch.sum(lamb3 * total3)
    loss4 = torch.sum(lamb4 * total4)

    loss = loss1 + loss2 + loss3 + loss4
    return loss, loss1, loss2, loss3, loss4


def resolvePath(path, base_dir):
    # Use the path as given if it exists (or contains a directory),
    # otherwise look for it inside base_dir.
    if os.path.isabs(path) or os.path.exists(path) or '/' in path:
        return path
    return os.path.join(base_dir, path)


def isNull(path):
    return path is None or path is False or str(path).strip() in ['', 'NULL', 'None', 'False']


def main(args):

    # set arguments
    data_path = args.data_dir
    input_path = args.input_dir
    res_dir = args.res_dir
    test_file = args.test_file
    cm_file = args.stoichiometry_matrix
    sc_imputation = args.sc_imputation
    fileName = args.output_flux_file
    balanceName = args.output_balance_file
    EPOCH = args.train_epoch
    grr_path = args.grr_path


    if EPOCH <= 0:
        raise NameError('EPOCH must greater than 1!')

    # choose cpu or gpu automatically
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # seed both expression noise and network initialization
    np.random.seed(SEED)

    print("Starting load data...")

    # read table of gene expression level
    geneExpr = pd.read_csv(
                resolvePath(test_file, input_path),
                index_col=0)
    geneExpr = geneExpr.astype(float)

    # Tissue Specificity :
    # for the top 10% of genes of each cell or cluster, multiply 4 with gene expression level
    # for the top 10-20% of genes of them, multiply 2 with gene expression level for default
    print('Applying tissue specificity')
    weightAlign = np.ones(geneExpr.shape[0])

    for _ in tissueSpecific:
        weightAlign[int(-(_[0]*len(weightAlign))): ] = _[1]

    geneExprVal = []
    for topCombi in range(geneExpr.shape[1]):
        singleColumn = np.array(geneExpr[geneExpr.columns[topCombi]].values)
        singleColRank = singleColumn.argsort()
        singleColWeight = np.ones(len(singleColumn))

        singleColWeight[singleColRank] = weightAlign

        geneExprVal.append(pd.DataFrame(singleColumn*singleColWeight))

    geneExprVal = pd.concat(geneExprVal, axis = 1)

    geneExprVal.index = geneExpr.index
    geneExprVal.columns = geneExpr.columns
    geneExpr.iloc[:, :] = geneExprVal.iloc[:, :]
    del geneExprVal

    geneExpr = geneExpr.T
    geneExpr = geneExpr * 1.0

    if sc_imputation == True:
        import magic  # only needed for imputation
        magic_operator = magic.MAGIC()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            geneExpr = magic_operator.fit_transform(geneExpr)

    if geneExpr.max().max() > 50:
        geneExpr = (geneExpr + 1).apply(np.log2)  

    geneExprSum = geneExpr.sum(axis=1)
    stand = geneExprSum.mean()
    geneExprScale = geneExprSum / stand
    geneExprScale = torch.FloatTensor(geneExprScale.values).to(device)
    
    BATCH_SIZE = geneExpr.shape[0]

    # load gene reaction rule
    with open(resolvePath(grr_path, data_path), 'r') as grrF:
        grr = grrF.read()
    grr = [_.strip('\r') for _ in grr.split('\n') if _.strip() != '']

    for _ in grr:
        if len(_.split(',')) != 3:
            raise ValueError('grr line must be "pathway,reaction,gene": ' + _)

    # create reaction pathway dictionary and sort gene reaction rules by reaction name
    rxnPath = {}
    rxnDict = {}
    for _ in grr:
        parts = _.split(',')
        pathway = parts[0]
        reaction = parts[1]
        
        # build reaction pathway dictionary
        if pathway in rxnPath:
            if reaction not in rxnPath[pathway]:
                rxnPath[pathway].append(reaction)
        else:
            rxnPath[pathway] = [reaction]
        
        # build reaction pathway dictionary (using reaction as key, genes as value)
        try:
            rxnDict[reaction].append(frozenset(parts[2].split(' and ')))
        except:
            rxnDict[reaction] = [frozenset(parts[2].split(' and '))]

    # set of all gene included in reactions
    tmpreactionGene = [_.split(',')[2].split(' and ') for _ in grr]
    reaction_gene_all = set(reduce(lambda grrTmpA, grrTmpB: grrTmpA + grrTmpB, tmpreactionGene, []))
    del tmpreactionGene


    # fix genes in both data and reaction rules
    # set virtual genes in reaction rules but not in data
    data_gene_all = set(geneExpr.columns)
    gene_overlap = list(data_gene_all.intersection(reaction_gene_all))
    virtual_gene = list(set(reaction_gene_all) - set(data_gene_all) - set(['']))

    if len(virtual_gene) > 0:
        virtual_df = np.zeros((geneExpr.shape[0], len(virtual_gene)))
        virtual_df = pd.DataFrame(virtual_df, index = list(geneExpr.index), columns = virtual_gene)
        gene_overlap = gene_overlap + virtual_gene
        
        geneExpr = pd.concat([geneExpr, virtual_df], axis = 1)

    gene_overlap.sort()

    cmMat = pd.read_csv(
            resolvePath(cm_file, data_path),
            sep=',',
            index_col = 0)

    rxnDir = cmMat.iloc[0, :]
    cmMat = cmMat.iloc[1:, :]

    rxns = list(cmMat.columns)
    cName = list(cmMat.index)

    # every reaction in the matrix needs a gene reaction rule
    missing_rxn = [_ for _ in rxns if _ not in rxnDict]
    if len(missing_rxn) > 0:
        raise ValueError('Reactions in matrix but not in grr: ' + ', '.join(missing_rxn))
    extra_rxn = [_ for _ in rxnDict if _ not in rxns]
    if len(extra_rxn) > 0:
        print('Warning: %d reactions in grr but not in matrix are ignored' % len(extra_rxn))

    print('Model: %d reactions, %d metabolites, pathways: %s' % (len(rxns), len(cName), ', '.join(rxnPath.keys())))
    if len(virtual_gene) > 0:
        print('Genes in grr but not in input (set to 0): ' + ', '.join(sorted(virtual_gene)))

    cmMat = cmMat.values
    cmMat = torch.FloatTensor(cmMat).to(device)
    
    print("Load data done.")
    
    print("Starting process data...")

    # extract overlap gene
    geneExpr = geneExpr[gene_overlap]
    gene_names = geneExpr.columns
    cell_names = geneExpr.index.astype(str)
    geneExprIndex = geneExpr.index.tolist()
    geneExprZero = geneExpr.copy()
    geneExprZero.iloc[:, :] = 0
    geneExprRepo = {frozenset(['']) : geneExprZero.copy()}

    n_rxn = len(rxns)
    n_genes = len(gene_names)
    n_cells = len(cell_names)
    n_comps = cmMat.shape[0]
    geneExprDf = pd.DataFrame(columns = ['reaction_gene'] + list(cell_names))

    tmp_geneExprDf = []
    tempMerged_prev = []
    

    seq_n_genes = [0]
    for i, rxn_name in enumerate(rxns):
        v = rxnDict[rxn_name]

        if v == [frozenset([''])]:
            temp = geneExprRepo[frozenset([''])].copy()
            seq_n_genes.append(seq_n_genes[-1])

        else:
            if frozenset(v) in geneExprRepo.keys():
                temp = geneExprRepo[frozenset(v)].copy()
                seq_n_genes.append(seq_n_genes[-1] + temp.shape[1])
            
            else:
                tempMerged = reduce(lambda grrTmpA, grrTmpB: list(grrTmpA) + list(grrTmpB), v, [])
                if len(v) == len(tempMerged):
                    temp = geneExpr.copy()
                    temp.loc[temp.index, [g for g in temp.columns if g not in tempMerged]] = 0

                else:
                    tmp_rule = [list(_) for _ in v]
                    temp = andRule(geneExpr.copy(), tmp_rule)

                seq_n_genes.append(seq_n_genes[-1] + len(tempMerged))

        geneExprRepo[frozenset(v)] = temp.copy()
        temp = temp.T            

        temp['reaction_gene'] = ['%02d_%s' % (i,g) for g in gene_names]
        tmp_geneExprDf.append(temp)


    geneExprDf = pd.concat(tmp_geneExprDf, ignore_index = True)

    exprRand = pd.DataFrame(np.random.uniform(0, 1e-1, size = (geneExprDf.shape[0], geneExprDf.shape[1] - 1)))
    exprRand.index = geneExprDf.index
    exprRand.columns = geneExprDf.columns[:-1]
    geneExprDf.iloc[:, :-1] += exprRand


    # Set Direction Constraints.
    # Reactions are grouped into forward only, bidirectional and reverse only.
    # Groups are based on Recon3D model.
    dirCons = []
    for _ in rxns:
        dirCons.append(rxnDir[_])

    dirCons = torch.FloatTensor(np.array(dirCons))
    dirFactor = (dirCons+0.1)/abs(dirCons+0.1)
    dirCons = torch.pow(dirCons, 2)
    cmMat = cmMat*dirFactor


    geneExprDf.index = geneExprDf['reaction_gene']
    geneExprDf.drop('reaction_gene', axis = 'columns', inplace = True)
    X = geneExprDf.values.T.astype(np.float32)
    X = torch.FloatTensor(X).to(device)

    balanceCons = torch.where(torch.where(cmMat != 0, 1, 0).sum(axis=1) > 1, 1, 0)
    balanceCons = balanceCons.reshape(len(balanceCons), 1)
    
    # prepare data for constraint of reaction variation based on gene
    df = geneExprDf
    df.index = [i.split('_')[0] for i in df.index]
    df.index = df.index.astype(int)   # mush change type to ensure correct order, T column name order change!
    rxn_scale = df.groupby(df.index).sum().T
    rxn_scale = torch.FloatTensor(rxn_scale.values)
    

    print("Process data done.")
    
# =============================================================================
    #NN
    torch.manual_seed(SEED)
    net = FLUX(X, n_rxn, f_in = n_genes, f_out = 1).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr = LEARN_RATE)

    dataSet = MyDataset(X, geneExprScale, rxn_scale)
    train_loader = torch.utils.data.DataLoader(dataset = dataSet, batch_size = BATCH_SIZE, shuffle = False)

# =============================================================================
    print("Starting train neural network...")
    start = time.time()
    loss_v = []
    net.train()
    for epoch in tqdm(range(EPOCH)):
        loss = np.zeros(5)
        for X_batch, X_scale_batch, r_scale_batch in train_loader:
            out_r_batch, out_c_batch = net(X_batch.float().to(device), n_rxn, n_genes, n_comps, cmMat)
            losses = myLoss(out_r_batch, out_c_batch,
                            lamb1 = LAMB_BA, lamb2 = LAMB_NG, lamb3 = LAMB_CELL, lamb4 = LAMB_RXN,
                            geneScale = X_scale_batch.float().to(device), reactionScale = r_scale_batch.float().to(device),
                            dirConstraints = dirCons, balConstraints = balanceCons)
            optimizer.zero_grad()
            losses[0].backward()
            optimizer.step()
            loss += [float(l) for l in losses]
        loss_v.append(loss)
    loss_v = np.array(loss_v)
    end = time.time()
    print("Training time: ", end - start)

    net.eval()
    with torch.no_grad():
        out_r, out_c = net(X, n_rxn, n_genes, n_comps, cmMat)
    setF = pd.DataFrame(out_r.cpu().numpy(), index = geneExprIndex, columns = rxns)
    setB = pd.DataFrame(out_c.cpu().numpy(), index = geneExprIndex, columns = cName)

# =============================================================================
    # save results
    timestr = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(res_dir, exist_ok=True)
    tag = timestr

    with open("./" + res_dir + "/lossValue_" + tag + ".txt", "w") as file_loss:
        for epoch, l in enumerate(loss_v):
            file_loss.write('epoch: %02d, loss1: %.8f, loss2: %.8f, loss3: %.8f, loss4: %.8f, loss: %.8f. \n'
                            % (epoch + 1, l[1], l[2], l[3], l[4], l[0]))

    plt.plot(loss_v[:, 0], '--')
    for j in range(1, 5):
        plt.plot(loss_v[:, j])
    plt.legend(['total', 'balance', 'negative', 'cellVar', 'reactionVar']);
    plt.savefig('./' + res_dir + '/loss_' + tag + ".png")
    plt.clf()

    if fileName == 'NULL':
        fileName = "./" + res_dir + "/flux_" + tag + ".csv"
    setF.to_csv(fileName)

    if balanceName == 'NULL':
        balanceName = "./" + res_dir + "/balance_" + tag + ".csv"
    setB.to_csv(balanceName)

    print("scFEA job finished. Check result in the desired output folder.")

    return


def parse_arguments(parser):


    parser.add_argument('--data_dir', type=str, default='data', metavar='<data_directory>',
                        help='The data directory for scFEA model files.')
    parser.add_argument('--input_dir', type=str, default='input', metavar='<input_directory>',
                        help='The data directory for single cell input data.')
    parser.add_argument('--res_dir', type=str, default='output', metavar='<data_directory>',
                        help='The data directory for result [output]. The output of scFEA includes two matrices, predicted metabolic flux and metabolites stress at single cell resolution.')
    parser.add_argument('--test_file', type=str, default='IgA_all.csv', 
                        help='The test SC file [input]. The input of scFEA is a single cell profile matrix, where row is gene and column is cell. Example datasets are provided in /data/ folder. The input can be raw counts or normalised counts. The logarithm would be performed if value larger than 30.')
    parser.add_argument('--stoichiometry_matrix', type=str, default='pyrimidine_matrix.csv', 
                        help='The table describes relationship between compounds and reactions. Each row is an intermediate metabolite and each column is metabolic reaction. For human model, please use cmMat_171.csv which is default. All candidate stoichiometry matrices are provided in /data/ folder.')
    parser.add_argument('--sc_imputation', type=eval, default='False', choices=[True, False],
                        help='Whether perform imputation for SC dataset (recommend set to <True> for 10x data).')
    parser.add_argument('--output_flux_file', type=str, default='NULL', 
                        help='User defined predicted flux file name.')
    parser.add_argument('--output_balance_file', type=str, default='NULL', 
                        help='User defined predicted balance file name.')
    parser.add_argument('--train_epoch', type=int, default=100, nargs='?',
                        help='User defined EPOCH (training iteration).')
    parser.add_argument('--grr_path', type=str, default='pyrimidine_grr_path.csv', metavar='<grr_file>',
                        help='Gene reaction rule file (pathway,reaction,gene) in data_dir.')

    
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='scFEA: A graph neural network model to estimate cell-wise metabolic flux using single cell RNA-seq data')
    args = parse_arguments(parser)

    startTime = time.time()
    main(args)
    endTime = time.time()
    print("Total time : ", endTime - startTime)
