import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.cluster import adjusted_mutual_info_score
dropT2D=True
if __name__=='__main__':
    refvec = pd.read_csv('./Vec/Shanghai/288/vec_total_823.csv',index_col=0)
    refvec = refvec.sort_values('index')
    refvec.index = refvec['index']
    refvec = refvec.iloc[:,:128]
    clusterkey = 'v2_dropT2D'
    reflabel = pd.read_csv('./Results/vec_total_823/Cluster_'+clusterkey+'.csv',index_col=0)
    if dropT2D:
        reflabel = reflabel[reflabel['cluster']!='Diabetes']
    refvec = refvec.loc[reflabel.index]
    reflabel = reflabel.loc[reflabel.index]
    refcluster = reflabel['cluster']

    #targetvec = pd.read_csv('./Vec/Shanghai/288/vec_total_823_mergevec.csv',index_col=0)
    targetvec = pd.read_csv('./Vec/Colas_emb/Colas_vec_mergevec.csv', index_col=0)

    #Summary = pd.read_csv('../Data/Summary_Shanghai.csv',index_col=0)
    Summary = pd.read_csv('../Data/Summary_Colas.csv', index_col=0)
    #allT2D = [x for x in targetvec.index if Summary.loc[x,'type']=='T2D']
    #targetvec = targetvec.loc[~targetvec.index.isin(allT2D)]

    resultsimi = cosine_similarity(np.array(refvec),np.array(targetvec))
    resultsimi = pd.DataFrame(resultsimi,index=refvec.index,columns=targetvec.index)
    result = pd.DataFrame(index=list(set(refcluster.values)),columns=targetvec.index,dtype='float64')
    for i in result.index:
        result.loc[i] = resultsimi[refcluster==i].mean(axis=0)
    result.T.to_csv('./TempResult.csv')
    result = pd.DataFrame({'cluster':result.idxmax()},index=targetvec.index)

    #temp = pd.DataFrame(index=allT2D,columns=result.columns)
    #temp['cluster']='Diabetes'
    #result = result.append(temp[['cluster']])
    result['label'] = Summary.loc[result.index]['type']
    #result['OGTT_Type'] = Summary.loc[result.index]['OGTT_Type']

    result.to_csv('./Results/Colas_vec/Cluster_Sample_'+clusterkey+'.csv')
    print(set(result['cluster']))

    typecluster = ['Normal', 'Pre_Ia', 'Pre_Ib', 'Pre_IIc', 'Pre_IIa', 'Pre_IIb', 'Diabetes']
    type = ['NGT', 'IGR', 'T2D']
    NumsCount = pd.DataFrame(index=typecluster, columns=type)
    for t1 in typecluster:
        for t2 in type:
            NumsCount.loc[t1, t2] = len(result[(result['cluster'] == t1) & (result['label'] == t2)])
    print(NumsCount)
    #print(adjusted_mutual_info_score(result['cluster'],result['label']))
    #print(adjusted_mutual_info_score(result[pd.notna(result['OGTT_Type'])]['cluster'],result[pd.notna(result['OGTT_Type'])]['OGTT_Type']))