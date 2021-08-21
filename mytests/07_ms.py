from copy import deepcopy
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy import set_printoptions
from typing import Tuple
from numpy.lib.shape_base import tile
set_printoptions(threshold=1000000000)

import time
def time_watcher(previous_time=None, print_key=""):
    current_time = time.time()
    if previous_time is None:
        print('time_watcher start')
    else:
        print('{}: {}'.format(print_key, current_time - previous_time))
    return current_time

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def plot_data(centroids, data, pred_centers, n_samples, label, pattern):
    max_label = np.max(label) + 2
    camp = get_cmap(max_label)
    colors_arr = []
    for i in range(len(label)):
        colors_arr.append(camp(label[i]))
    plt.scatter(data[:,0], data[:,1], color=colors_arr)
    plt.savefig('data_{}.png'.format(pattern))
    plt.show()
    # colour = plt.cm.rainbow(np.linspace(0,1,len(centroids)))
    for i, centroid in enumerate(centroids):
        # samples = X[i*n_samples:(i+1)*n_samples]
        # plt.scatter(samples[:,0], samples[:,1], c=colour[i], s=1)
        plt.plot(centroid[0], centroid[1], markersize=10, marker="x", color='k', mew=5)

    for centroid in pred_centers:
        plt.plot(centroid[0], centroid[1], markersize=5, marker="x", color='m', mew=2)

    plt.savefig('gt_{}.png'.format(pattern))
    plt.clf()

import torch
def gaussian(d:torch.FloatTensor, bw:float) -> torch.FloatTensor:
    return torch.exp(-0.5*((d/bw))**2) / (bw*math.sqrt(2*math.pi))

def flat(d:torch.FloatTensor, bw:float) -> torch.FloatTensor:
    res: torch.BoolTensor = d < bw
    return res.to(dtype=d.dtype)

class TorchMeanShift:
    """https://github.com/fastai/courses/blob/master/deeplearning2/meanshift.ipynb
    """

    supported_kernels = {
        'gaussian': gaussian,
        'flat': flat
    }

    def __init__(self, bandwidth, max_iter=300, kernel='flat') -> None:
        self.bandwidth = bandwidth
        self.max_iter = max_iter

        if kernel in self.supported_kernels:
            self.kernel = self.supported_kernels[kernel]
        else:
            raise NotImplementedError('Supported kernels are {}, actually {}'.format(self.supported_kernels.keys(), kernel))

        self.labels_ = None
        self.cluster_all = True
        self.n_iter_ = 0

    @staticmethod
    def _get_pairwise_distances(a,b) -> torch.FloatTensor:
        """Get a pairwise distance matrix.
        """
        return torch.sqrt(((a.unsqueeze(0) - b.unsqueeze(1))**2).sum(2)).T

    @staticmethod
    def _get_radius_nn_mask(X1, X2, radius, including_myself=True) -> torch.BoolTensor:
        """Get mask of pairwise distance matrix for radius nearest neighbors.
        Args:
            X1 (torch.tensor) : input (N1, C)
            X2 (torch.tensor) : input (N2, C)
            radius (float): radius
            including_myself (bool): In case of including_myself=False and X1=X2, return results not including myself in radius nearest neighbors. But, when there is no neighborhood, nan may be mixed into radius_nn_mean (return).
        Return:
            radius_nn_mask (torch.tensor): mask of radius nearest neighbors (N1, N2)

        Examples:
            X = torch.arange(12, dtype=torch.float32).reshape(4,3)
            mask = MeanshiftG._get_radius_nn_mask(X, X, 5)
        """
        dist = TorchMeanShift._get_pairwise_distances(X1, X2)
        radius_nn_mask:torch.tensor = dist < radius

        # If X1.shape=X2.shape and including_myself is True, this function does not include myself in raidus NN for the mean calculation.
        # But, when there is no neighborhood, nan may be mixed.
        if not including_myself and X1 == X2:
            N, C = X1.shape
            diag_mask = torch.diag(torch.ones(N)).to(dtype=torch.bool)
            radius_nn_mask[diag_mask] = False

        return radius_nn_mask

    @staticmethod
    def _get_radius_nn_mean(X:torch.tensor, radius:float, including_myself=True) -> Tuple[torch.FloatTensor, torch.BoolTensor]:
        """Get mean of nearest neighbors in radius.
        Args:
            X (torch.tensor) : inputs (N, C)
            radius (float): radius
            including_myself (bool): In case of False, return results not including myself in radius nearest neighbors. But, when there is no neighborhood, nan may be mixed into radius_nn_mean (return).
        Return:
            radius_nn_mean (torch.tensor): mean of radius nearest neighbors (N)
            radius_nn_mask (torch.tensor): mask of radius nearest neighbors (N, N)

        Examples:
            X = torch.arange(12, dtype=torch.float32).reshape(4,3)
            tms = TorchMeanShift(5, max_iter=5)
            mean_data, mask = tms._mean_radius_nn(X)
        """

        # Get radius NN mask
        radius_nn_mask = TorchMeanShift._get_radius_nn_mask(X, X, radius, including_myself)

        # Get radius NN mean (including myself)
        N, C = X.shape
        X = torch.tile(X[:, None, :], (1, N, 1))
        X = X.transpose(0, 1)
        X[radius_nn_mask == False] = 0
        radius_nn_mean:torch.FloatTensor = X.sum(1) / radius_nn_mask.sum(-1)[:, None]

        return radius_nn_mean, radius_nn_mask
    
    @staticmethod
    def _t2n(torch_tensor:torch.Tensor) -> np.ndarray:
        """torch.Tensor to numpy.ndarray
        """
        return torch_tensor.detach().cpu().numpy()

    def _create_labels(self, X:torch.tensor, original_X:torch.tensor):
        device = X.device

        # 0: 0.0008115768432617188
        # 1: 0.020339250564575195
        # 2: 0.009784221649169922
        # 3: 0.0003573894500732422
        # 4: 0.00017213821411132812
        # 5: 0.0005629062652587891
        # 6: 0.00017404556274414062

        t = time_watcher()
        # get all_res (sklearn)
        radius_nn_mean, radius_nn_mask = self._get_radius_nn_mean(X, self.bandwidth)
        t = time_watcher(t, '0')
        num_nn = torch.sum(radius_nn_mask, dim=1)
        t = time_watcher(t, '1')

        seeds = original_X
        center_intensity_dict = {}

        for i in range(len(seeds)):
            if num_nn[i] > 1:  # i.e. len(points_within) > 0
                center_intensity_dict[tuple(radius_nn_mean[i].cpu().numpy().tolist())] = num_nn[i]
        t = time_watcher(t, '2')

        sorted_by_intensity = sorted(center_intensity_dict.items(),
                                     key=lambda tup: (tup[1], tup[0]),
                                     reverse=True)
        sorted_centers = torch.tensor([tup[0] for tup in sorted_by_intensity], device=device)
        t = time_watcher(t, '3')

        radius_nn_mask = TorchMeanShift._get_radius_nn_mask(sorted_centers, sorted_centers, self.bandwidth)
        unique = torch.ones(len(sorted_centers), dtype=bool, device=device)
        t = time_watcher(t, '4')

        for i in range(len(sorted_centers)):
            if unique[i]:
                neighbor_idxs = radius_nn_mask[i]
                unique[neighbor_idxs] = 0
                unique[i] = 1  # leave the current point as unique
        cluster_centers = sorted_centers[unique]
        t = time_watcher(t, '5')

        # ASSIGN LABELS: a point belongs to the cluster that it is closest to
        dist = self._get_pairwise_distances(original_X, cluster_centers)
        idxs = torch.argmin(dist, dim=1)
        labels = torch.zeros(len(original_X), dtype=int)
        t = time_watcher(t, '6')

        if self.cluster_all:
            labels = idxs.flatten()
        else:
            raise NotImplementedError()
            # labels.fill(-1)
            # bool_selector = dist.flatten() <= self.bandwidth
            # labels[bool_selector] = idxs.flatten()[bool_selector]

        self.cluster_centers_ = TorchMeanShift._t2n(cluster_centers)
        self.labels_ = TorchMeanShift._t2n(labels)

    def fit(self, X:torch.FloatTensor):
        original_X = deepcopy(X)
        stop_thresh = 1e-3 * self.bandwidth

        with torch.no_grad():
            # X = torch.FloatTensor(np.copy(X)).cuda()
            start = time.time()
            for it in range(self.max_iter):
                weight = self.kernel(self._get_pairwise_distances(X, X), self.bandwidth)
                num = (weight[:, :, None] * X).sum(1)
                X_new = num / weight.sum(1)[:, None]
                # check convergence
                shift = torch.abs(X_new - X).sum()/torch.abs(original_X.sum())
                X = X_new
                self.n_iter_ += 1
                if shift < stop_thresh:
                    break
            print(time.time()-start)
            start = time.time()
            self._create_labels(X, original_X)
            print(time.time()-start)

    def predict(self):
        raise NotImplementedError()


n_clusters=6
n_samples =250
centroids = np.random.uniform(-35, 35, (n_clusters, 2))
slices = [np.random.multivariate_normal(centroids[i], np.diag([5., 5.]), n_samples)
           for i in range(n_clusters)]
data = np.concatenate(slices).astype(np.float32)

tms = TorchMeanShift(4, max_iter=500, kernel='flat')
tms.fit(torch.from_numpy(data).to(0))
labels = tms.labels_
centers = tms.cluster_centers_
print('gpu iter: {}'.format(tms.n_iter_))
plot_data(centroids, data, centers, n_samples, labels, 'gpu')

from sklearn.cluster import MeanShift
ms = MeanShift(bandwidth=4)
ms.fit(data)
labels = ms.labels_
centers = ms.cluster_centers_
print('cpu iter: {}'.format(ms.n_iter_))
plot_data(centroids, data, centers, n_samples, labels, 'cpu')

