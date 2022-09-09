import seaborn as sns
import matplotlib.pyplot as plt

def correlogram(data,col,h_neg=240,h_pos=10,figsize=(10, 5)):
    
    cor=data[col].corr()
    f, ax = plt.subplots(figsize=figsize)

    cmap = sns.diverging_palette(h_neg, h_neg, as_cmap=True)

    sns.heatmap(cor, cmap=cmap,vmin=-1,vmax=1)
    plt.show()
    
def pairplot(data,col,corner=True):
    
    sns.pairplot(data[col],corner=corner)
    plt.show()
    
def barplot1(data,col,hue=None,rows=2,cols=2,figsize=(10,5),xlabel_size=15,xticks_size=10,hspace=0.4,wspace=None,rotation=45):
    
    fig, ax = plt.subplots(rows,cols,figsize=figsize)
    fig.subplots_adjust(hspace=hspace,wspace=wspace)

    for c, a in zip(col,ax.flatten()):
        
        data_col = data.groupby(c).count().iloc[:,0]/data.shape[0]
        (data_col).plot(kind='bar',ax=a)
        a.set_xticklabels(data_col.index,rotation=rotation,fontdict={'fontsize':xticks_size})
        a.set_xlabel(c,fontsize=xlabel_size)

    plt.show()
    
    
def boxplot1(data,y,col,rows=2,cols=2,figsize=(10,5),label_size=15,xticks_size=10,hspace=0.4,wspace=None,rotation=45):
    
    fig, ax = plt.subplots(rows,cols,figsize=figsize)
    fig.subplots_adjust(hspace=hspace,wspace=wspace)

    for c, a in zip(col,ax.flatten()):
        
        sns.boxplot(y=y,x=c,hue=c,orient='v',data=data,ax=a)
        a.legend().remove()
        a.set_xticklabels(a.get_xticklabels(),rotation=rotation,fontdict={'fontsize':xticks_size})
        a.set_ylabel(y,fontsize=label_size)
        a.set_xlabel(c,fontsize=label_size)

    plt.show()
