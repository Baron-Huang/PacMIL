import torch

def PCA_svd(X, k, center=True):
  n = X.size()[0]
  ones = torch.ones(n).view([n,1])
  h = ((1/n) * torch.mm(ones, ones.t())) if center else torch.zeros(n*n).view([n,n])
  H = torch.eye(n) - h
  H = H.cuda()
  X_center =  torch.mm(H.double(), X.double())
  u, s, v = torch.svd(X_center)
  components  = v[:k].t()
  #explained_variance = torch.mul(s[:k], s[:k])/(n-1)
  return components



torch.pca_lowrank()