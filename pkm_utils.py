import numpy as np
from scipy.io import loadmat
from pykalman import KalmanFilter
# import matplotlib.pyplot as plt
# import multiprocessing
from joblib import Parallel, delayed

#%% definitions
def make_F(dim_obs: int, Delta_t: float):
  """
  This matrix is used to model the evolution of the system's state (e.g. positions and velocities) over time;
  dim_obs: size of a state transition matrix;
  makes F assuming a constant velocity model;
  """
  blocked_row_1 = np.hstack([np.eye(dim_obs), np.eye(dim_obs)*Delta_t])
  blocked_row_2 = np.hstack([np.eye(dim_obs)*0., np.eye(dim_obs)])
  F = np.vstack([blocked_row_1, blocked_row_2])
  assert(F.shape == (dim_obs*2, dim_obs*2))
  return F

def make_Q(dim_obs: int, Delta_t: float, sigma_eta: float):
    """
    This computes the covariance of transition matrix;
    makes Q assuming a constant velocity model
    """
    blocked_row_1 = np.hstack([np.eye(dim_obs)*0., np.eye(dim_obs)*0.])
    blocked_row_2 = np.hstack([np.eye(dim_obs)*0., np.eye(dim_obs)*(Delta_t**2)*(sigma_eta**2)])
    Q = np.vstack([blocked_row_1, blocked_row_2])
    assert(Q.shape == (dim_obs*2, dim_obs*2))
    return Q

def make_H(dim_obs: int):
    """
    compute the size of the observation matrix
    """
    H =  np.hstack([np.eye(dim_obs), np.eye(dim_obs) * 0.])
    assert(H.shape == (dim_obs, dim_obs*2))
    return H

def make_model(data, sigma_eta = 10., sigma_v = .003, initial_var_diag = 0.01, Delta_t=1/50): # 50 is the frame rate
    '''
    This is a wrapper function to implement Kalman Filter on behavioral keypoint predictions
    :param data: output of behavioral keypoint prediction (trackPosture_slp_XL.ipynb).
    :param Delta_t: 1/FrameRate
    :return: filtered output
    '''
    dim_obs = data.shape[1] # define the size of observation space (i.e. tracking in how many dimensions)
    dim_latent = dim_obs*2 # state that the size of hidden states is 2 times larger than that of observation space
    effective_acceleration_noise = Delta_t ** 2 * (sigma_eta ** 2)
    aux_diag_noise = effective_acceleration_noise / 1e4
    init_mean = np.zeros(dim_latent)
    init_mean[:dim_obs] = data[0]
    init_cov = np.concatenate([np.ones(int(dim_latent / 2)), 2 * np.ones(int(dim_latent / 2))])  # double the noise on velocity
    init_cov = np.diag(init_cov * initial_var_diag)

    kf = KalmanFilter(initial_state_mean=init_mean,
                      initial_state_covariance=init_cov,
                      transition_matrices=make_F(dim_obs, Delta_t),
                      transition_covariance=make_Q(dim_obs, Delta_t, sigma_eta) + np.eye(
        dim_latent) * aux_diag_noise,
                      observation_matrices=make_H(dim_obs),
                      observation_covariance=np.eye(dim_obs) * sigma_v ** 2)
    return kf

def kf_smooth_preds(preds, reproj, repThresh=15):
    nT = preds.shape[0]
    num_bodyparts = preds.shape[1]
    num_coords = preds.shape[2]
    smoothed_pos = np.full_like(preds, fill_value=np.NaN)
    smoothed_vel = np.full_like(preds, fill_value=np.NaN)

    for nPart in range(num_bodyparts):
        print('Smoothing part #{0}'.format(nPart))
        raw_data = np.ma.zeros((nT, num_coords))
        raw_mask = reproj[:,nPart] > repThresh
        for i in range(num_coords):
            raw_data[:,i] = np.ma.array(preds[:,nPart,i], mask=raw_mask) # points with reprojection error larger than 15 are treated as unreliable and masked
        kf = make_model(raw_data)
        (smoothed_state_means, smoothed_state_covariances) = kf.smooth(raw_data)
        smoothed_pos[:,nPart,:] = smoothed_state_means[:,:num_coords]
        smoothed_vel[:, nPart, :] = smoothed_state_means[:, num_coords:]
    return smoothed_pos, smoothed_vel

def kf_inner(preds, reproj, partNum, repThresh=15):
    print('Smoothing part #{0}'.format(partNum))
    nT = preds.shape[0]
    num_coords=preds.shape[1]
    raw_data = np.ma.zeros((nT, num_coords))
    raw_mask = reproj > repThresh
    for i in range(num_coords):
        raw_data[:,i] = np.ma.array(preds[:,i], mask=raw_mask)
    kf = make_model(raw_data)
    (smoothed_state_means, smoothed_state_covariances) = kf.smooth(raw_data)
    smoothed_pos = smoothed_state_means[:, :num_coords]
    smoothed_vel = smoothed_state_means[:, num_coords:]
    return smoothed_pos, smoothed_vel

def kf_outer(preds, reproj, repThresh):
    num_bodyparts = preds.shape[1]
    res = Parallel(n_jobs=num_bodyparts)(delayed(kf_inner)
            (preds[:, i], reproj[:, i], i, repThresh) for i in range(num_bodyparts))

    smoothed_pos = np.full_like(preds, fill_value=np.NaN)
    smoothed_vel = np.full_like(preds, fill_value=np.NaN)
    for i in range(num_bodyparts):
        smoothed_pos[:, i, :] = res[i][0]
        smoothed_vel[:, i, :] = res[i][1]
    return smoothed_pos, smoothed_vel

#%%
if __name__ == '__main__':
    # load file
    thisDir = "/media/selmaan/Locker/Selmaan/Birds/LIM97/LIM97_220221_121913"
    fn = loadmat(thisDir + '/posture_2stage_5_13.mat')
    # fn = loadmat(thisDir + '\posture_2stage_4_12b.mat')

    repThresh = 12
    preds = np.concatenate([fn['posture_preds'],fn['com_preds']],axis=1)
    reproj = np.concatenate([fn['posture_reproj'],fn['com_reproj']],axis=1)
    smPos, smVel = kf_outer(preds, reproj, repThresh)

    from scipy.io import savemat
    savemat(thisDir + '/tmpSmooth.mat',
            {'smPos':smPos, 'smVel':smVel})

#%%
# args = []
# for i in range(num_bodyparts):
#     args.append((preds[:,i], reproj[:,i], i, repThresh))

# split body parts into 2 batches of 9 parallel jobs
# if __name__ == '__main__':
#     with multiprocessing.Pool(num_bodyparts) as p:
#         res = p.starmap(kf_inner, args)

#
#
# def f(x):
#     print(x)
#
# if __name__ == '__main__':
#     with multiprocessing.Pool(4) as p:
#         p.map(f, range(10))
#
# #%%
# # from scipy.io import savemat
# # (smP, smV) =  kf_smooth_preds(fn['posture_preds'],fn['posture_reproj'])
# # savemat('Z:\Selmaan\Birds\LIM99\LIM99_210520_110404\\tmpSmooth.mat',{'smPos':smP, 'smVel':smV})
#
#
# # %% Setup
# nT = 425000
# repThresh = 15
# # data = fn['com_preds']
# # reproj = fn['com_reproj']
# data = fn['posture_preds'][:, 9:]
# reproj = fn['posture_reproj'][:, 9:]
# num_bodyparts = data.shape[1]
# num_coords = 3
# # convert t x k x 3d into a nT x k*3 matrix, with 3d for k=1, then 3d for k=2, and so on
# raw_data = np.ma.zeros((nT, num_bodyparts * num_coords))
# for i in range(num_bodyparts):
#     raw_mask = reproj[:nT, i] > repThresh
#     for j in range(num_coords):
#         raw_data[:nT, i * num_coords + j] = np.ma.array(data[:nT, i, j], mask=raw_mask)
#
# Delta_t = 1 / 60
# dim_obs = num_bodyparts * num_coords
# dim_latent = dim_obs * 2
#
# #%% Tuneable hyperparameters
# sigma_eta = 10. # the noise of the unknown zero-mean acceleration, smaller values -> slower motion
# sigma_v = .003 # observation noise, larger values -> more smoothing
# initial_var_diag = 0.1**2
# effective_acceleration_noise = Delta_t**2 * (sigma_eta**2)
# aux_diag_noise = effective_acceleration_noise / 1e4 # a diagonal correction to make Q nondegenerate. dividing by 10 to get an order of magnitude smaller than the actual effective noise
#
#
# #%% set up the model using hyperparameters, smooth, and plot
#
# param_dict = {}
# param_dict["F"] = make_F(dim_obs, Delta_t)
# param_dict["Q"] = make_Q(dim_obs, Delta_t, sigma_eta) + np.eye(dim_latent) * aux_diag_noise # np.array([[0., 0.], [0., Delta_t**2 * (sigma_eta**2)]]) + np.eye(dim_latent) * aux_diag_noise
# param_dict["H"] = make_H(dim_obs) # np.array([[1., 0.]])
# param_dict["R"] = np.eye(dim_obs) * sigma_v**2
# # note the specifications of the inits below (double initial noise on the unknown velocity )
# param_dict["init_mu"] = np.concatenate([raw_data[0,:], np.zeros(int(dim_latent/2))]) # initialize assuming
# diagonal_init_cov_mat = np.diag(np.concatenate([np.ones(int(dim_latent/2)), 2*np.ones(int(dim_latent/2))])) # double the noise on velocity
# param_dict["init_cov"] = diagonal_init_cov_mat * initial_var_diag
# kf = KalmanFilter(initial_state_mean= param_dict["init_mu"],
#                   initial_state_covariance= param_dict["init_cov"],
#                 transition_matrices = param_dict["F"],
#                   transition_covariance= param_dict["Q"],
#                   observation_matrices = param_dict["H"],
#                  observation_covariance= param_dict["R"])
#
# (smoothed_state_means, smoothed_state_covariances) = kf.smooth(raw_data)
#
# #%%
# # # X
# # plt.plot(raw_data[:,0::3])
# # plt.plot(smoothed_state_means[:,0:dim_obs:3], 'k');
# # plt.title('X'),
# # plt.show(),
# #
# # # Y
# # plt.plot(raw_data[:,1::3])
# # plt.plot(smoothed_state_means[:,1:dim_obs:3], 'k');
# # plt.title('Y'),
# # plt.show(),
# #
# # # Z
# # plt.plot(raw_data[:,2::3])
# # plt.plot(smoothed_state_means[:,2:dim_obs:3], 'k');
# # plt.title('Z'),
# # plt.show(),
#
# #%% replot after optimizing with EM
#
# kf.em(raw_data, n_iter=3)
# (smoothed_state_means, smoothed_state_covariances) = kf.smooth(raw_data)
#
# # X
# plt.plot(raw_data[:,0::3])
# plt.plot(smoothed_state_means[:,0:dim_obs:3], 'k');
# plt.title('X'),
# plt.show(),
#
# # Y
# plt.plot(raw_data[:,1::3])
# plt.plot(smoothed_state_means[:,1:dim_obs:3], 'k');
# plt.title('Y'),
# plt.show(),
#
# # Z
# plt.plot(raw_data[:,2::3])
# plt.plot(smoothed_state_means[:,2:dim_obs:3], 'k');
# plt.title('Z'),
# plt.show(),