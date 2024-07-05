import numpy as np

import torch

from aux import flat_data
from rff_mmd_approx import weights_sphere, weights_rahimi_recht, rff_sphere, rff_rahimi_recht
from slicedKL import slicedKLclass


def data_label_embedding_no_reduce(data, rff_params, mmd_type):
  data_embedding = rff_sphere(data, rff_params) if mmd_type == 'sphere' else rff_rahimi_recht(data, rff_params)
  return data_embedding

#mmd util start
def euclidsq(x, y):
    return torch.pow(torch.cdist(x, y), 2)

def prepare(x_de, x_nu):
    return euclidsq(x_de, x_de), euclidsq(x_de, x_nu), euclidsq(x_nu, x_nu)

def gaussian_gramian(esq, σ):
    return torch.exp(torch.div(-esq, 2 * σ**2))

USE_SOLVE = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def kmm_ratios(Kdede, Kdenu, λ):
    n_de, n_nu = Kdenu.shape
    if λ > 0:
        A = Kdede + λ * torch.eye(n_de).to(device)
    else:
        A = Kdede
    # Equivalent implement based on 1) solver and 2) matrix inversion
    if USE_SOLVE:
        B = torch.sum(Kdenu, 1, keepdim=True)
        return (n_de / n_nu) * torch.linalg.solve(A,B)
    else:
        B = Kdenu
        return torch.matmul(torch.matmul(torch.inverse(A), B), torch.ones(n_nu, 1).to(device))

eps_ratio = 0.0001
clip_ratio = True

def estimate_ratio_compute_mmd(x_de, x_nu, σs=[]):
    
    dsq_dede, dsq_denu, dsq_nunu = prepare(x_de, x_nu)
    
    if len(σs) == 0:
        with torch.no_grad():
        # A heuristic is to use the median of pairwise distances as σ, suggested by Sugiyama's book
            sigma = torch.sqrt(
                torch.median(
                    torch.cat([dsq_dede, dsq_denu, dsq_nunu], 1)
                )#median
            )
            c = 2
            σs = sigma*torch.as_tensor([1/c,0.333/c,0.2/c,5/c,3/c,.1/c],device=device)
            #print("heuristic sigma: ", sigma)
            
    is_first = True
    ratio = None
    #mmdsq = None
    for σ in σs:
        Kdede = gaussian_gramian(dsq_dede, σ)
        Kdenu = gaussian_gramian(dsq_denu, σ)
        Knunu = gaussian_gramian(dsq_nunu, σ)
        if is_first:
            ratio = kmm_ratios(Kdede, Kdenu, eps_ratio)
            #mmdsq = mmdsq_of(Kdede, Kdenu, Knunu)
            is_first = False
        else:
            ratio += kmm_ratios(Kdede, Kdenu, eps_ratio)
            #mmdsq += mmdsq_of(Kdede, Kdenu, Knunu)
    
    ratio = ratio / len(σs)
    #ratio = torch.relu(ratio) if clip_ratio else ratio
    #mmd = torch.sqrt(torch.relu(mmdsq))
    #return ratio, mmd
    return ratio

def sliced_kl_kmm_diff_priv(first_samples,
                                second_samples,
                                thetas, #n_slice * slice_dim * d
                                p=1,                                
                                device='cuda',
                                sigma_proj=1,
                                sigma_noise = 1,
                                noise_samples=None,
                                n_slice=40,
                                slice_dim=2
                                ):
    # first samples are the data to protect
    # second samples are the data_fake
    
    #first_samples, second_samples = make_sample_size_equal(first_samples, second_samples)

    #dim = second_samples.size(1)
    nb_sample = second_samples.size(0)
    thetas = thetas.to(device)
    sigma_noise = sigma_noise.to(device)
    #projections = rand_projections_diff_priv(dim, num_projections,sigma_proj)
    #projections = projections.to(device)
    noise2 = torch.randn((n_slice,nb_sample,slice_dim),device=device)*sigma_noise
    if noise_samples is not None:
        noise = noise_samples * sigma_noise.to(device)
    else:
        noise = torch.randn((n_slice,nb_sample,slice_dim))*sigma_noise.to(device)

    first_projections = torch.matmul(first_samples,torch.transpose(thetas, 1,2)) + noise
    second_projections = torch.matmul(second_samples,torch.transpose(thetas, 1,2)) + noise2

    #ratio=torch.zeros_like(first_projections,device=device)
    
    
    #find good kernel width to share across the slices
    with torch.no_grad():
        # A heuristic is to use the median of pairwise distances as σ, suggested by Sugiyama's book
        diff = torch.linalg.vector_norm(first_projections,dim=2) # - second_projections)**2 #n_slice * batch * slice_dim
        sigma = torch.sqrt(torch.median(diff**2))
        sigma = 20
        #print("heuristic sigma: ", sigma)
        #sigmaprime = torch.sqrt(torch.median(second_projections**2))
        #print((sigma - sigmaprime)/sigma)
        
        
        c = 2
        # σs = sigma*torch.as_tensor([1/c,0.333/c,0.2/c,5/c,3/c,.1/c],device=device)
        σs = sigma*torch.as_tensor([1/c,0.333/c,3/c],device=device)
        # σs = sigma*torch.as_tensor([1/c],device=device)
    
    def ratio_func(first,second):
        return estimate_ratio_compute_mmd(first,second,σs = σs)
    vec_mmd=torch.vmap(ratio_func,in_dims=0,out_dims=0)
    
    ratio = vec_mmd(first_projections,second_projections).view(n_slice,-1)
    #for i in range(first_projections.shape[1]):
    #    ratio[:,i] = estimate_ratio_compute_mmd(first_projections[:,i].view(-1,1), second_projections[:,i].view(-1,1), []).squeeze() 
        
    #print(ratio[:,0].shape)
    #print(ratio.shape)
    #print(ratio[:,0])
    
    epsilon=torch.full(ratio.size(),1e-10,device=device)
    ratio = torch.maximum(ratio,epsilon)
    #print(torch.log(ratio))
    # kl = torch.mean(ratio*torch.log(ratio))
    # return kl
    chi_squared = torch.mean(torch.pow(ratio-1,2))
    return chi_squared
#mmd util end

def sliced_wasserstein_distance_diff_priv(first_samples,
                                second_samples,
                                thetas,
                                p=1,                                
                                device='cuda',
                                sigma_proj=1,
                                sigma_noise = 1,
                                noise_samples=None
                                ):
    # first samples are the data to protect
    # second samples are the data_fake
    
    #first_samples, second_samples = make_sample_size_equal(first_samples, second_samples)

    #dim = second_samples.size(1)
    nb_sample = second_samples.size(0)
    #projections = rand_projections_diff_priv(dim, num_projections,sigma_proj)
    #projections = projections.to(device)
    noise2 = torch.randn((nb_sample,thetas.shape[0])).to(device)*sigma_noise
    # noise2 = noise2.to(device)
    if noise_samples is not None:
        noise = noise_samples.to(device) * sigma_noise
        # noise = noise.to(device)  
    else:
        noise = torch.randn((nb_sample,thetas.shape[0])).to(device)*sigma_noise
        # noise = noise.to(device)    
    #print(first_samples.shape)
    #print(second_samples.shape)
    first_projections = torch.matmul(first_samples,torch.transpose(thetas[:,0,:], 0,1)) + noise 
    second_projections = torch.matmul(second_samples,torch.transpose(thetas[:,0,:], 0,1)) + noise2 
    # print(first_projections.shape, second_projections.shape)
    wasserstein_distance = torch.abs((torch.sort(first_projections.transpose(0, 1), dim=1)[0] -
                                      torch.sort(second_projections.transpose(0, 1), dim=1)[0]))
    wasserstein_distance = torch.pow(torch.mean(torch.pow(wasserstein_distance, p), dim=1), 1. / p) # averaging the sorted distance
    return torch.pow(torch.pow(wasserstein_distance, p).mean(), 1. / p)  # averaging over the random direction

def calc_sigma(epsilon, d, n_KL_slices,n_KL_slice_dim, delta=None, data_size=None):
    assert delta is not None or data_size is not None, "either delta or data_size has to be provided"

    ####################FIGURE OUT WHAT NOISE TO ADD
    epsilon = epsilon #- preprocessor_eps
    if delta is None:
        delta =  1 / np.sqrt(data_size) #* np.sqrt(48842)
    
    mprime = n_KL_slice_dim * n_KL_slices
    
    #print(data_dim)
    #print(epsilon)
    # First, try approx-opt alpha expression to guess sigma
    aa = mprime/d
    bb = 2 * np.sqrt(mprime * np.log(1/delta) / d)
    cc = epsilon
    sigma_propose = ( (-bb + np.sqrt(bb**2 + 4 * aa *cc))  /(2*aa) )**(-1) # Will always be real and positive

    # Iterative optimization
    iters = 10
    for i in range(iters):
        # Compute implied approximate-opt alpha 
        alpha_star = 1 + np.sqrt(sigma_propose**2 * d * np.log(1/delta) / mprime)

        # Check if implied alpha is outside allowed range and pick closest allowed
        if (alpha_star**2 - alpha_star) > d * sigma_propose**2 / 2:
            # quadratic formula, won't be imaginary or unambiguous
            alpha_star = 0.5 * (1 + np.sqrt(1 + 2 * d * sigma_propose**2))
        # Recompute sigma in case alpha changed, using exact formula for epsilon
        val = 2 * d * (epsilon - np.log(1/delta) / (alpha_star - 1) ) / (mprime * alpha_star  )
        while val <= 0:
            if ((1.2*alpha_star)**2 - 1.2*alpha_star) > d * sigma_propose**2:
                val = 0.001
                print('WARNING: unable to find a valid sigma to achieve specified epsilon, delta combination. Using a large sigma.')
                break
            else:
                alpha_star *= 1.2
                #print('WARNING: unable to find a valid sigma to achieve specified epsilon, delta. Using a large sigma.')
                val = 2 * d * (epsilon - np.log(1/delta) / (alpha_star - 1) ) / (mprime * alpha_star  ) #.01
        val2 = (alpha_star**2 - alpha_star) / d
        sigma_propose = np.sqrt(  (1 / val) + val2 ) #Automatically satisfies constraint if val > 0
    sigma = sigma_propose
    noise = sigma
    epsilon_actual = mprime * alpha_star / (2 * sigma**2 * (d - (alpha_star**2 - alpha_star)*sigma**(-2))) + np.log(1/delta)/(alpha_star - 1)
    print('User specified (epsilon, delta)=(' + str(epsilon) + ',' + str(delta) + '); Chosen sigma = ' + str(sigma) + '; Actual epsilon = ' + str(epsilon_actual))
    
    sigma_noise = sigma
    ################################################
    return sigma_noise

def get_sliced_losses(train_loader, d, d_rff, rff_sigma, mmd_type, n_KL_slices, n_KL_slice_dim, epsilon, device):
    d_enc = d
    d = d_rff
    assert d_rff % 2 == 0
    assert isinstance(rff_sigma, str)
    rff_sigma = [float(sig) for sig in rff_sigma.split(',')]
    if mmd_type == 'sphere':
        w_freq = weights_sphere(d_rff, d_enc, rff_sigma, device)
    else:
        w_freq = weights_rahimi_recht(d_rff, d_enc, rff_sigma, device)
    data_size = len(train_loader.dataset)

    kldiv = slicedKLclass(d,n_KL_slices,n_KL_slice_dim,device).to(device)
    # noise_data_kmm = torch.randn(n_KL_slices, data_size, n_KL_slice_dim).to(device)
    noise_data_kmm = torch.randn(data_size, n_KL_slices * n_KL_slice_dim).to(device)
    # delta = 1e-5
    sigma_noise = calc_sigma(epsilon, d, n_KL_slices, n_KL_slice_dim, data_size=data_size)
    maxi_maxi_norm_singleton = [None]

    rff_params = w_freq
    def minibatch_loss(data_enc, labels, gen_enc, gen_labels, X_noise_kmm=None):
        c1 = None
        real_cat = data_enc
        fakeact = gen_enc
        real_cat = data_label_embedding_no_reduce(real_cat, rff_params, mmd_type)
        fakeact = data_label_embedding_no_reduce(fakeact, rff_params, mmd_type)
        if X_noise_kmm is None:
            # X_noise_kmm = torch.randn(n_KL_slices, real_cat.shape[0], n_KL_slice_dim).to(device)
            X_noise_kmm = torch.randn(real_cat.shape[0], n_KL_slices * n_KL_slice_dim).to(device)

        if c1 is not None:
            fakey = torch.cat([fakeact, c1], dim=1)
            #YY = fakey.view(fakey.shape[0], -1)
        else:
            fakey = fakeact
            #YY = fakeact.view(fakeact.shape[0],-1)
            
        #if epoch > epoch_to_start_align:
        with torch.no_grad():
            maxi_norm = torch.sqrt(torch.max(torch.sum(real_cat.view(real_cat.shape[0], -1)**2,dim=1))).to(device)
            if maxi_maxi_norm_singleton[0] is None or maxi_norm > maxi_maxi_norm_singleton[0]:
                maxi_maxi_norm_singleton[0] = maxi_norm
            maxi_maxi_norm = maxi_maxi_norm_singleton[0]

        source_features_norm = torch.div(real_cat,1) #(2*maxi_norm))
        target_features_norm = torch.div(fakey,1) #(2*maxi_norm))
        
        # kl = sliced_kl_kmm_diff_priv(source_features_norm.view(source_features_norm.shape[0], -1), 
        #                             target_features_norm.view(target_features_norm.shape[0], -1),
        #                             kldiv.thetas,
        #                             2,
        #                             device,
        #                             sigma_noise=sigma_noise * 2 * maxi_maxi_norm, 
        #                             noise_samples=X_noise_kmm,
        #                             n_slice=n_KL_slices, 
        #                             slice_dim=n_KL_slice_dim)
        # wd_clf = 1
        # loss_g = wd_clf * kl

        loss_g = sliced_wasserstein_distance_diff_priv(
            source_features_norm.view(source_features_norm.shape[0], -1), 
            target_features_norm.view(target_features_norm.shape[0], -1),
            kldiv.thetas,
            2,
            device,
            sigma_noise=sigma_noise * 2 * maxi_maxi_norm, 
            noise_samples=X_noise_kmm)

        return loss_g
    
    train_loader_iter_singleton = [iter(train_loader)] # wrap as a lst for easy change in place
    sinlge_labels = None

    def single_release_loss(gen_enc, gen_labels):
        try:
            (data, labels), (ix,) = next(train_loader_iter_singleton[0])
        except StopIteration:
            train_loader_iter_singleton[0] = iter(train_loader)
            (data, labels), (ix,) = next(train_loader_iter_singleton[0])
        data = data.to(device)
        sinlge_data_enc = flat_data(data, labels, device, n_labels=10, add_label=False)
        # return minibatch_loss(sinlge_data_enc, sinlge_labels, gen_enc, gen_labels, X_noise_kmm=noise_data_kmm[:,ix,:])
        return minibatch_loss(sinlge_data_enc, sinlge_labels, gen_enc, gen_labels, X_noise_kmm=noise_data_kmm[ix,:])

    return single_release_loss, minibatch_loss