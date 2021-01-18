import sys
import click
import pickle
import librosa
import numpy as np
from sklearn.cluster import KMeans

@click.group()
def cli_train():
    pass

@click.group()
def cli_test():
    pass

def extract_mfcc(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc, order=1)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc = np.vstack((mfcc, mfcc_delta, mfcc_delta))
    mfcc = np.swapaxes(mfcc, 0, 1)
    return mfcc

@cli_train.command()
@click.argument('train_list_fname', type=click.Path(exists=True))
@click.argument('codebooks_fname', type=click.Path())
@click.option('-nc', 'n_clusters', type=int, 
            default=32, show_default=True, help='Number of clusters')
def train(train_list_fname, codebooks_fname, n_clusters):
    '''Command on training'''
    train_list = open(train_list_fname).read().splitlines()
    codebooks = []
    for file_path in train_list:
        y, sr = librosa.load(file_path)
        y, _ = librosa.effects.trim(y)
        mfcc = extract_mfcc(y, sr)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(mfcc)
        codebooks.append(kmeans)
    pickle.dump(codebooks, open(codebooks_fname, 'wb'))
    
@cli_test.command()
@click.argument('test_list_fname', type=click.Path(exists=True))
@click.argument('codebooks_fname', type=click.Path(exists=True))
@click.option('-o', 'output_fname', type=click.Path(), help='Output file name')
def test(test_list_fname, codebooks_fname, output_fname):
    '''Command on test'''
    test_list = open(test_list_fname).read().splitlines()
    codebooks = pickle.load(open(codebooks_fname, 'rb'))
    if output_fname:
        sys.stdout = open(output_fname, 'w')
    for file_path in test_list:
        y, sr = librosa.load(file_path)
        y, _ = librosa.effects.trim(y)
        mfcc = extract_mfcc(y, sr)
        pred_lbls = []
        for codebook in codebooks:
            pred_lbls.append(codebook.predict(mfcc))
        dists = []
        for j in range(mfcc.shape[0]):
            dists.append([])
            for k in range(len(codebooks)):
                center = codebooks[k].cluster_centers_[pred_lbls[k][j]]
                dists[j].append(np.linalg.norm(center - mfcc[j]))
        dists = np.array(dists)
        speakers = np.argmin(dists, axis=1)
        counts = np.bincount(speakers)
        print(file_path + " " + str(np.argmax(counts)))

cli = click.CommandCollection(sources=[cli_train, cli_test])

if __name__ == '__main__':
    cli()