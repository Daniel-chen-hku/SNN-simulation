from matplotlib import colorbar
import numpy as np
from data_class import RC_ECG
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

path = '/home/hegan/ECG/Disease/'
cd_set = np.load(path+'sample_CD10.npz')
sample_cd = cd_set['train_data']
hyp_set = np.load(path+'sample_HYP10.npz')
sample_hyp = hyp_set['train_data']
mi_set = np.load(path+'sample_MI10.npz')
sample_mi = mi_set['train_data']
norm_set = np.load(path+'sample_NORM10.npz')
sample_norm = norm_set['train_data']
sttc_set = np.load(path+'sample_STTC10.npz')
sample_sttc = sttc_set['train_data']
rc = RC_ECG(k=10,a=2)
norm = rc.rc_function(sample_norm - sample_norm.min())
cd = rc.rc_function(sample_cd - sample_cd.min())
hyp = rc.rc_function(sample_hyp - sample_hyp.min())
mi = rc.rc_function(sample_mi - sample_mi.min())
sttc = rc.rc_function(sample_sttc - sample_sttc.min())
norm_dim_reduced = PCA(n_components=2).fit_transform(norm)
cd_dim_reduced = PCA(n_components=2).fit_transform(cd)
hyp_dim_reduced = PCA(n_components=2).fit_transform(hyp)
mi_dim_reduced = PCA(n_components=2).fit_transform(mi)
sttc_dim_reduced = PCA(n_components=2).fit_transform(sttc)
plt.plot(norm_dim_reduced[:,0],norm_dim_reduced[:,1],'o','r')
plt.plot(cd_dim_reduced[:,0],norm_dim_reduced[:,1],'o','b')
plt.plot(hyp_dim_reduced[:,0],norm_dim_reduced[:,1],'o','g')
plt.plot(mi_dim_reduced[:,0],norm_dim_reduced[:,1],'o','y')
plt.plot(sttc_dim_reduced[:,0],norm_dim_reduced[:,1],'o','k')

# norm_dim_reduced = PCA(n_components=3).fit_transform(norm)
# cd_dim_reduced = PCA(n_components=3).fit_transform(cd)
# hyp_dim_reduced = PCA(n_components=3).fit_transform(hyp)
# mi_dim_reduced = PCA(n_components=3).fit_transform(mi)
# sttc_dim_reduced = PCA(n_components=3).fit_transform(sttc)
# fig1 = plt.figure()
# ax1 = plt.axes(projection='3d')
# ax1.scatter3D(norm_dim_reduced[:,0],norm_dim_reduced[:,1],c='b')
# ax1.scatter3D(cd_dim_reduced[:,0],norm_dim_reduced[:,1],c='k')
# ax1.scatter3D(hyp_dim_reduced[:,0],norm_dim_reduced[:,1],c='r')
# ax1.scatter3D(mi_dim_reduced[:,0],norm_dim_reduced[:,1],c='g')
# ax1.scatter3D(sttc_dim_reduced[:,0],norm_dim_reduced[:,1],c='y')

