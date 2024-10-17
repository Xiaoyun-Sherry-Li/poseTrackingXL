"""
MIT License (MIT)
Copyright (c) FALL 2016, Jahdiel Alvarez
Author: Jahdiel Alvarez
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
Based on Scipy's cookbook:
http://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
"""

import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

#%%
class PySBA:
    """Python class for Simple Bundle Adjustment"""

    def __init__(self, cameraArray, points3D, points2D, cameraIndices, point2DIndices, pointWeights=None):
        """Intializes all the class attributes and instance variables.
            Write the specifications for each variable:
            cameraArray with shape (n_cameras, 11) contains initial estimates of parameters for all cameras.
                    First 3 components in each row form a rotation vector,
                    next 3 components form a translation vector,
                    then a focal distance and two distortion parameters,
                    then x,y image center coordinates
            points_3d with shape (n_points, 3)
                    contains initial estimates of point coordinates in the world frame.
            camera_ind with shape (n_observations,)
                    contains indices of cameras (from 0 to n_cameras - 1) involved in each observation.
            point_ind with shape (n_observations,)
                    contains indices of points (from 0 to n_points - 1) involved in each observation.
            points_2d with shape (n_observations, 2)
                    contains measured 2-D coordinates of points projected on images in each observations.
            pointWeights with shape (n_observations, )
                    contains cost function weights for each observation point.
        """
        self.cameraArray = cameraArray
        self.points3D = points3D
        self.points2D = points2D

        self.cameraIndices = cameraIndices
        self.point2DIndices = point2DIndices
        if pointWeights is None:
            pointWeights = np.full_like(point2DIndices, 1)
        self.pointWeights = pointWeights.reshape((-1, 1))

    """ Utils for converting 3D world points to 2D camera points """
    def rotate(self, points, rot_vecs):
        """Rotate points by given rotation vectors.
        Rodrigues' rotation formula is used.
        """
        theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
        with np.errstate(invalid='ignore'):
            v = rot_vecs / theta
            v = np.nan_to_num(v)
        dot = np.sum(points * v, axis=1)[:, np.newaxis]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v


    def project(self, points, cameraArray):
        """Convert 3-D points to 2-D by projecting onto images."""
        points_proj = self.rotate(points, cameraArray[:, :3])
        points_proj += cameraArray[:, 3:6]
        points_proj = points_proj[:, :2] / points_proj[:, 2, np.newaxis]
        # points_proj -= cameraArray[:, 9:] / 1778
        f = cameraArray[:, 6]
        k1 = cameraArray[:, 7]
        k2 = cameraArray[:, 8]
        n = np.sum(points_proj ** 2, axis=1)
        r = 1 + k1 * n + k2 * n ** 2
        points_proj *= (r * f)[:, np.newaxis]
        points_proj += cameraArray[:, 9:]
        return points_proj


    """ To jointly optimize 3D points and camera params """
    def fun(self, params, n_cameras, n_points,\
                camera_indices, point_indices, points_2d, pointWeights):
        """
        Compute residuals.
        Optimizing over the `params` variable, here both camera parameters and 3-D coordinates.
        """
        nCamParams = 11
        camera_params = params[:n_cameras * nCamParams].reshape((n_cameras, nCamParams))
        points_3d = params[n_cameras * nCamParams:].reshape((n_points, 3))
        points_proj = self.project(points_3d[point_indices], camera_params[camera_indices])
        weighted_residual = pointWeights * (points_proj - points_2d)
        
        return weighted_residual.ravel()

    def jac_sparsity_fun(self, numCameras, numPoints, cameraIndices, pointIndices):
        """
        Define the sparsity structure of the Jacobian matrix for least squares to speed up computation.
        """
        nCamParams = 11
        m = cameraIndices.size * 2
        n = numCameras * nCamParams + numPoints * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(cameraIndices.size)
        for s in range(nCamParams):
            A[2 * i, cameraIndices * nCamParams + s] = 1
            A[2 * i + 1, cameraIndices * nCamParams + s] = 1

        for s in range(3):
            A[2 * i, numCameras * nCamParams + pointIndices * 3 + s] = 1
            A[2 * i + 1, numCameras * nCamParams + pointIndices * 3 + s] = 1

        return A


    def optimizedParams(self, params, n_cameras, n_points):
        """
        Retrieve camera parameters and 3-D coordinates.
        """
        nCamParams = 11
        camera_params = params[:n_cameras * nCamParams].reshape((n_cameras, nCamParams))
        points_3d = params[n_cameras * nCamParams:].reshape((n_points, 3))

        return camera_params, points_3d


    def bundleAdjust(self):
        """
        Returns the bundle adjusted parameters, in this case the optimized:
            camera parameters: rotation and translation vectors, focal distance, skew, and offset.
            3D points: best guess for the 3D world coordinates of the set of points.
        """
        numCameras = self.cameraArray.shape[0]
        numPoints = self.points3D.shape[0]

        x0 = np.hstack((self.cameraArray.ravel(), self.points3D.ravel()))
        A = self.jac_sparsity_fun(numCameras, numPoints, self.cameraIndices, self.point2DIndices)

        #res = least_squares(self.fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
        #                    args=(numCameras, numPoints, self.cameraIndices, self.point2DIndices, self.points2D))
        res = least_squares(self.fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-6, method='trf', jac='3-point',
                            args=(numCameras, numPoints, self.cameraIndices, self.point2DIndices, self.points2D, self.pointWeights))

        camera_params, points_3d = self.optimizedParams(res.x, numCameras, numPoints)
        self.cameraArray = camera_params
        self.points3D = points_3d

        return res

    def getResiduals(self):
        """Gets residuals given current camera parameters and 3d locations"""
        numCameras = self.cameraArray.shape[0]
        numPoints = self.points3D.shape[0]
        x0 = np.hstack((self.cameraArray.ravel(), self.points3D.ravel()))
        f0 = self.fun(x0, numCameras, numPoints, self.cameraIndices, self.point2DIndices, self.points2D, np.full_like(self.point2DIndices, 1))
        return f0

    """ To optimize just 3D points, holding camera params fixed """
    def fun_nocam(self, params, camera_params, n_points,\
                    camera_indices, point_indices, points_2d, pointWeights):
        """Compute residuals. The `params` var contains 3-D coordinates only.
        """
        points_3d = params.reshape((n_points, 3))
        points_proj = self.project(points_3d[point_indices], camera_params[camera_indices])
        weighted_residual = pointWeights * (points_proj - points_2d)
        
        return weighted_residual.ravel()

    def jac_sparsity_nocam(self, numPoints, pointIndices):
        m = pointIndices.size * 2
        n = numPoints * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(pointIndices.size)
        for s in range(3):
            A[2 * i, pointIndices * 3 + s] = 1
            A[2 * i + 1, pointIndices * 3 + s] = 1

        return A

    def bundleAdjust_nocam(self):
        """ Returns the optimized 3d positions given current camera parameters,
        without adjusting the camera parameters themselves. """
        numPoints = self.points3D.shape[0]
        camera_params = self.cameraArray

        x0 = self.points3D.ravel()
        A = self.jac_sparsity_nocam(numPoints, self.point2DIndices)
        res = least_squares(self.fun_nocam, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-8, method='trf', jac='3-point',
                            args=(camera_params, numPoints, self.cameraIndices, self.point2DIndices, self.points2D, self.pointWeights))
        self.points3D = res.x.reshape((numPoints, 3))

        return res

    """ To optimize just the camera extrinsics using known 3D pts and holding the instrinsics fixed """
    def fun_just_extrinsics(self, params, cam_intrinsics, points_3d, n_cameras, \
                            camera_indices, point_indices, points_2d, pointWeights):
        """
        Compute residuals, minimizing wrt 'params.'
        Here, params is only the camera parameters.
        """
        nCamParams = 11
        n_extrinsics = 6
        cam_extrinsics = params.reshape((n_cameras, n_extrinsics))
        camera_params = np.column_stack([cam_extrinsics, cam_intrinsics])

        points_proj = self.project(points_3d[point_indices], camera_params[camera_indices])
        weighted_residual = pointWeights * (points_proj - points_2d)
        
        return weighted_residual.ravel()

    def jac_sparsity_just_extrinsics(self, n_cameras, cameraIndices):
        n_extrinsics = 6
        m = cameraIndices.size * 2
        n = n_cameras * n_extrinsics
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(cameraIndices.size)
        for s in range(n_extrinsics):
            A[2 * i, cameraIndices * n_extrinsics + s] = 1
            A[2 * i + 1, cameraIndices * n_extrinsics + s] = 1

        return A

    def bundleAdjust_just_extrinsics(self):
        """ Returns the optimized camera extrinsic params given known 3d positions,
        without adjusting the 3d positions themselves and holding the instrinsic camera params fixed. """
        n_extrinsics = 6
        n_cameras = self.cameraArray.shape[0]
        points_3d = self.points3D
        cam_intrinsics = self.cameraArray[:, n_extrinsics:]

        x0 = self.cameraArray[:, :n_extrinsics].ravel()
        A = self.jac_sparsity_just_extrinsics(n_cameras, self.cameraIndices)
        res = least_squares(self.fun_just_extrinsics, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-8, method='trf', jac='3-point',
                            args=(cam_intrinsics, points_3d, n_cameras, self.cameraIndices, self.point2DIndices, self.points2D, self.pointWeights))
        
        cam_extrinsics = res.x.reshape((n_cameras, n_extrinsics))
        self.cameraArray = np.column_stack([cam_extrinsics, cam_intrinsics])

        return res

    """ To optimize just the camera params, holding 3D points fixed """
    def fun_justcam(self, params, points_3d, n_cameras,\
                    camera_indices, point_indices, points_2d, pointWeights):
        """
        Compute residuals, minimizing wrt 'params.'
        Here, params is only the camera parameters.
        """
        nCamParams = 11
        camera_params = params.reshape((n_cameras, nCamParams))
        points_proj = self.project(points_3d[point_indices], camera_params[camera_indices])
        weighted_residual = pointWeights * (points_proj - points_2d)
        
        return weighted_residual.ravel()

    def jac_sparsity_justcam(self, numCameras, cameraIndices):
        nCamParams = 11
        m = cameraIndices.size * 2
        n = numCameras * nCamParams
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(cameraIndices.size)
        for s in range(nCamParams):
            A[2 * i, cameraIndices * nCamParams + s] = 1
            A[2 * i + 1, cameraIndices * nCamParams + s] = 1

        return A

    def bundleAdjust_justcam(self):
        """ Returns the optimized camera parameters given known 3d positions,
        without adjusting the 3d positions themselves. """
        numCameras = self.cameraArray.shape[0]
        points_3d = self.points3D

        x0 = self.cameraArray.ravel()
        A = self.jac_sparsity_justcam(numCameras, self.cameraIndices)
        res = least_squares(self.fun_justcam, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-8, method='trf', jac='3-point',
                            args=(points_3d, numCameras, self.cameraIndices, self.point2DIndices, self.points2D, self.pointWeights))
        self.cameraArray = res.x.reshape((n_cameras, nCamParams))

        return res

    """ To optimize 3D points and camera params using cameras w/ shared properties """
    def fun_sharedcam(self, params, n_cameras, n_points, \
                        camera_indices, point_indices, points_2d, pointWeights):
        """Compute residuals.
        `params` contains camera parameters and 3-D coordinates.
        """
        nCamIntrinsic = 3
        nCamExtrinsic = 6
        nCamCentroid = 2
        nCamUnique = nCamExtrinsic + nCamCentroid
        nCamParams = n_cameras * nCamUnique + nCamIntrinsic

        cam_shared_intrinsic = params[:nCamIntrinsic]
        camera_extrinsic = params[nCamIntrinsic:nCamIntrinsic+n_cameras*nCamExtrinsic].reshape((n_cameras, nCamExtrinsic))
        camera_centroid = params[nCamIntrinsic+n_cameras*nCamExtrinsic : nCamParams].reshape((n_cameras, nCamCentroid))
        camera_params = np.concatenate((camera_extrinsic, np.tile(cam_shared_intrinsic, (n_cameras,1)), camera_centroid), axis=1)

        points_3d = params[nCamParams:].reshape((n_points, 3))
        points_proj = self.project(points_3d[point_indices], camera_params[camera_indices])
        weighted_residual = pointWeights * (points_proj - points_2d)
        return weighted_residual.ravel()

    def bundle_adjustment_sparsity_sharedcam(self, numCameras, numPoints, cameraIndices, pointIndices):
        m = cameraIndices.size * 2
        nCamIntrinsic = 3
        nCamExtrinsic = 6
        nCamCentroid = 2
        nCamParams = numCameras * nCamExtrinsic + numCameras * nCamCentroid + nCamIntrinsic
        n = nCamParams + numPoints * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(cameraIndices.size)
        A[2*i, 0:nCamIntrinsic] = 1
        A[2*i + 1, 0:nCamIntrinsic] = 1
        for s in range(nCamExtrinsic):
            A[2 * i, nCamIntrinsic + cameraIndices * nCamExtrinsic + s] = 1
            A[2 * i + 1, nCamIntrinsic + cameraIndices * nCamExtrinsic + s] = 1
        for s in range(nCamCentroid):
            A[2 * i, nCamIntrinsic + numCameras*nCamExtrinsic + cameraIndices * nCamCentroid + s] = 1
            A[2 * i + 1, nCamIntrinsic + numCameras*nCamExtrinsic + cameraIndices * nCamCentroid + s] = 1

        for s in range(3):
            A[2 * i, nCamParams + pointIndices * 3 + s] = 1
            A[2 * i + 1, nCamParams + pointIndices * 3 + s] = 1

        return A

    def ls_bounds_sharedcam(self, numCameras, numPoints):
        '''
        Defines bounds for various parameters to prevent least squares from going off the rails.
        These should be redefined depending on what is reasonable for one's set-up.
        TODO - update to match well with cam array post updating just extrinsics
        '''
        # shared intrinsics
        intrinsics_low = np.asarray([1300, -0.1, -0.1])
        intrinsics_high = np.asarray([2300, 0.1, 0.1])

        # extrinsics
        rot_1_low = np.asarray([0.7, 0.7, 1.4, 1.4])
        rot_1_high = np.asarray([1.0, 1.0, 2.0, 2.0])
        rot_2_low = np.asarray([1.8, -2.2, -0.9, 0.6])
        rot_2_high = np.asarray([2.2, -1.8, -0.6, 0.9])
        rot_3_low = np.asarray([-1.8, 1.4, 0.4, -0.8])
        rot_3_high = np.asarray([-1.4, 1.8, 0.8, -0.4])
        trans_12_low = np.full((numCameras, 2), -0.25)
        trans_12_high = np.full((numCameras, 2), 0.25)
        trans_3_low = np.full(numCameras, 1.3)
        trans_3_high = np.full(numCameras, 3.3)
        extrinsics_low = np.column_stack((rot_1_low, rot_2_low, rot_3_low, \
                                            trans_12_low, trans_3_low))
        extrinsics_low = extrinsics_low.ravel()
        extrinsics_high = np.column_stack((rot_1_high, rot_2_high, rot_3_high, \
                                            trans_12_high, trans_3_high))
        extrinsics_high = extrinsics_high.ravel()

        # camera centroids
        centroid_blue_low = np.asarray([950, 300])
        centroid_blue_high = np.asarray([1050, 400])
        centroid_others_low = np.full((numCameras-1, 2), [978, 310])
        centroid_others_high = np.full((numCameras-1, 2), [1078, 410])
        centroids_low = np.row_stack((centroid_others_low, centroid_blue_low))
        centroids_low = centroids_low.ravel()
        centroids_high = np.row_stack((centroid_others_high, centroid_blue_high))
        centroids_high = centroids_high.ravel()

        # 3D points
        xy_pts_low = np.full((numPoints, 2), -1.2)
        xy_pts_high = np.full((numPoints, 2), 1.2)
        z_pts_low = np.full(numPoints, -0.25)
        z_pts_high = np.full(numPoints, 0.25)
        pts_low = np.column_stack((xy_pts_low, z_pts_low)).ravel()
        pts_high = np.column_stack((xy_pts_high, z_pts_high)).ravel()

        # params bounds
        lower_bounds = np.hstack((intrinsics_low, extrinsics_low, centroids_low, pts_low))
        upper_bounds = np.hstack((intrinsics_high, extrinsics_high, centroids_high, pts_high))

        return (lower_bounds, upper_bounds)

    def bundleAdjust_sharedcam(self):
        """ Returns the bundle adjusted parameters, in this case the optimized
         rotation and translation vectors. """
        numCameras = self.cameraArray.shape[0]
        numPoints = self.points3D.shape[0]

        nCamIntrinsic = 3
        nCamExtrinsic = 6
        nCamCentroid = 2
        nCamUnique = nCamExtrinsic + nCamCentroid
        nCamParams = numCameras * nCamUnique + nCamIntrinsic

        camera_shared_intrinsic = np.mean(self.cameraArray[:, 6:9], axis=0).ravel()
        camera_extrinsic = self.cameraArray[:,:6].ravel()
        camera_centroids = self.cameraArray[:,9:].ravel()

        # set the independent variable and sparsity matrix
        x0 = np.hstack((camera_shared_intrinsic, camera_extrinsic, camera_centroids, self.points3D.ravel()))
        A = self.bundle_adjustment_sparsity_sharedcam(numCameras, numPoints, self.cameraIndices, self.point2DIndices)
        # x0_bounds = self.ls_bounds_sharedcam(numCameras, numPoints) # impose bounds on the params to keep things in check

        res = least_squares(self.fun_sharedcam, x0, jac_sparsity=A, # bounds=x0_bounds, 
                            verbose=2, x_scale='jac', ftol=1e-8, method='trf', jac='3-point',
                            args=(numCameras, numPoints, self.cameraIndices, self.point2DIndices, self.points2D, self.pointWeights))

        cam_shared_intrinsic = res.x[:nCamIntrinsic]
        camera_extrinsic = res.x[nCamIntrinsic:nCamIntrinsic + numCameras * nCamExtrinsic].reshape((numCameras, nCamExtrinsic))
        camera_centroid = res.x[nCamIntrinsic+numCameras*nCamExtrinsic : nCamParams].reshape((numCameras, nCamCentroid))
        camera_params = np.concatenate((camera_extrinsic, np.tile(cam_shared_intrinsic, (numCameras, 1)), camera_centroid), axis=1)
        points_3d = res.x[nCamParams:].reshape((numPoints, 3))
        self.cameraArray = camera_params
        self.points3D = points_3d

        return res


"""
Utils
"""
def convertParams(camParams):
    allParams = np.full((len(camParams), 11), np.NaN)
    for nCam in range(len(camParams)):
        p = camParams[nCam][0]
        f = p['K'][0,0]/2 + p['K'][1,1]/2
        r = -R.from_matrix(p['r']).as_rotvec()
        t = p['t']
        c = p['K'][2,0:2]
        d = p['RDistort']
        allParams[nCam,:] = np.hstack((r,t,f,d,c))
    return allParams

def DAParams(camParam): # XL created this to substitute the optCamArray form from pySBA with DA's matlab struct form
    K = np.array(camParam['K']) #camParams {1,1}, which is the first cam, will be [0,0] in python 
    r = np.array(camParam['r'])
    t = np.array(camParam['t'])
    d = np.array(camParam['RDistort'])
    # Clean up each array
    K = K[0,0]
    r = r[0,0]
    t = t[0,0]
    d = d[0,0]
    return {'K': K, 'R':r, 't':t, 'd':d}

def unconvertParams(camParamVec):
    thisK = np.full((3, 3), 0)
    thisK[0, 0] = camParamVec[6]
    thisK[1,1] = camParamVec[6]
    thisK[2,2] = 1
    thisK[2,:2] = camParamVec[9:]
    r = R.from_rotvec(-camParamVec[:3]).as_matrix()
    t = camParamVec[3:6]
    d = camParamVec[7:9]
    return {'K': thisK, 'R':r, 't':t, 'd':d}

def getCameraArray(file_path,
                    camera_ids=['red_cam', 'yellow_cam', 'green_cam', 'blue_cam'],
                    load_opt_array=False, opt_file_name='opt_cam_array.npy'):
    '''
    Params
    ------
    file_path : path to the camera array file (either init or optimized)
    camera_ids: list of camera names
    load_opt_array : if True, loads a previously saved optimized camera array
        If loading the optimized array, provide the file name.

    Returns
    -------
    camera_array : array of camera parameters; shape (n_camera, n_params)
    cam_array_fields : list of field names for each parameter

    Camera parameters are:
        Extrinsics
        ----------
        A 3D rotation vector that rotates the world coordinate axes into camera coordinate axes; array of floats, shape (3, )
        A 3D translation vector that translates the world origin to the camera origin; array of floats, shape (3, )

        Intrinsics
        ----------
        Focal distance in pixels; float
        Distortion params; array of floats, shape (2, )
        Principal point offsets (x, y); array of ints, shape (2, )

    These are initially estimated empirically (see il_rig_control/arena_alignment/init_cam_extrinsics)
    and OneNote notes (Camera Calibration).

    They are optimized during calibration and can be subsequently updated.
    '''
    n_cams = len(camera_ids)

    if load_opt_array:
        camera_array = np.load(f'{file_path}{opt_file_name}')
    else:
        camera_array = np.full((n_cams, 11), np.NaN)
        for i, cam in enumerate(camera_ids):
            camera_array[i] = np.load(f'{file_path}{cam}_array.npy')

    cam_array_fields = [
                        'rot_1', 'rot_2', 'rot_3',
                        'trans_1', 'trans_2', 'trans_3',
                        'focal dist', 'distort_1', 'distort_2',
                        'pt_x', 'pt_y'
                        ]

    return camera_array, cam_array_fields

def getCameraArray_SC(allCameras=['lBack', 'lFront', 'lTop', 'rBack', 'rFront', 'rTop']):
    '''
    Selmaan's estimated params.
    '''
    # Camera parameters are 3 rotation angles, 3 translations, 1 focal distance, 2 distortion params, and x,y principal points
    # Following notes outlined in evernote, 'bundle adjustment', later updated using optimized values
    camMatDict = {
        'lBack': np.array([0.72, -1.85, 1.73, 0.011, 0.144, 1.36, 1779, -0.021, -0.026, 1408, 704]),
        'lFront': np.array([1.88, -.63, .77, -0.041, .099, 1.41, 1779, -0.021, -0.026, 1408, 704]),
        'lTop': np.array([1.93, -1.77, 0.84, -.017, 0.076, 1.72, 1779, -0.021, -0.026, 1408, 848]),
        'rBack': np.array([0.79, 2.02, -1.77, 0.036, 0.1088, 1.37, 1779, -0.021, -0.026, 1408, 704]),
        'rFront': np.array([1.91, .75, -.69, 0.038, 0.1055, 1.38, 1779, -0.021, -0.026, 1408, 704]),
        'rTop': np.array([1.95, 1.899, -0.82, 0.0397, 0.0234, 1.73, 1779, -0.021, -0.026, 1408, 848]),
    }
    cameraArray = np.full((len(allCameras), 11), np.NaN)
    for i, e in enumerate(allCameras):
        cameraArray[i,:] = camMatDict[e]

    cam_array_fields = [
                        'rot_1', 'rot_2', 'rot_3',
                        'trans_1', 'trans_2', 'trans_3',
                        'focal dist', 'distort_1', 'distort_2',
                        'pt_x', 'pt_y'
                        ]

    return cameraArray, cam_array_fields