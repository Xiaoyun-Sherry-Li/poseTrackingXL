    ''' Functions to compute the residuals '''
    def fun(self, params, n_cameras, n_points, \
            camera_indices, point_indices, points_2d, pointWeights):
        """
        Compute residuals.
        `params` contains camera parameters and 3-D coordinates.
        """
        nCamParams = 11
        camera_params = params[:n_cameras * nCamParams].reshape((n_cameras, nCamParams))
        points_3d = params[n_cameras * nCamParams:].reshape((n_points, 3))
        points_proj = self.project(points_3d[point_indices], camera_params[camera_indices])
        weighted_residual = pointWeights * (points_proj - points_2d)
        
        return weighted_residual.ravel()

    def fun_nocam(self, params, camera_params, n_points, \
                    camera_indices, point_indices, points_2d, pointWeights):
        """Compute residuals.
        `params` contains 3-D coordinates only.
        """
        points_3d = params.reshape((n_points, 3))
        points_proj = self.project(points_3d[point_indices], camera_params[camera_indices])
        weighted_residual = pointWeights * (points_proj - points_2d)
        
        return weighted_residual.ravel()

    def fun_justcam(self, params, points_3d, n_cameras, \
                    camera_indices, point_indices, points_2d, pointWeights)
        """
        Compute residuals, minimizing wrt 'params.'
        Here, params is only the camera parameters.
        """
        nCamParams = 11
        camera_params = params.reshape((n_cameras, nCamParams))
        points_proj = self.project(points_3d[point_indices], camera_params[camera_indices])
        weighted_residual = pointWeights * (points_proj - points_2d)
        
        return weighted_residual.ravel()  

    def fun_just_extrinsics(self, params, cam_intrinsics, points_3d, n_cameras, \
                            camera_indices, point_indices, points_2d, pointWeights):
        """
        Compute residuals, minimizing wrt 'params.'
        Here, params is only the camera extrinsics.
        """
        nCamParams = 11
        n_extrinsics = 6
        cam_extrinsics = params.reshape((n_cameras, n_extrinsics))
        camera_params = np.column_stack([cam_extrinsics, cam_intrinsics])

        points_proj = self.project(points_3d[point_indices], camera_params[camera_indices])
        weighted_residual = pointWeights * (points_proj - points_2d)
        
        return weighted_residual.ravel()


    ''' Functions to define the sparsity structure of the Jacobian matrix for least squares '''
    def jac_sparsity_fun(self, n_cameras, numPoints, cameraIndices, pointIndices):
        nCamParams = 11
        m = cameraIndices.size * 2
        n = n_cameras * nCamParams + numPoints * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(cameraIndices.size)
        for s in range(nCamParams):
            A[2 * i, cameraIndices * nCamParams + s] = 1
            A[2 * i + 1, cameraIndices * nCamParams + s] = 1

        for s in range(3):
            A[2 * i, n_cameras * nCamParams + pointIndices * 3 + s] = 1
            A[2 * i + 1, n_cameras * nCamParams + pointIndices * 3 + s] = 1

        return A

    def jac_sparsity_nocam(self, numPoints, pointIndices):
        m = pointIndices.size * 2
        n = numPoints * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(pointIndices.size)
        for s in range(3):
            A[2 * i, pointIndices * 3 + s] = 1
            A[2 * i + 1, pointIndices * 3 + s] = 1

        return A

    def jac_sparsity_justcam(self, n_cameras, cameraIndices):
        nCamParams = 11
        m = cameraIndices.size * 2
        n = n_cameras * nCamParams
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(cameraIndices.size)
        for s in range(nCamParams):
            A[2 * i, cameraIndices * nCamParams + s] = 1
            A[2 * i + 1, cameraIndices * nCamParams + s] = 1

        return A

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


    ''' Functions to optimize the bundle adjusted parameters (3D coordinates and/or camera params) '''
    def bundleAdjust(self):
        """ Returns the bundle adjusted parameters, in this case the optimized
         rotation and translation vectors. """
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

    def bundleAdjust_justcam(self):
        """ Returns the optimized camera parameters given known 3d positions,
        without adjusting the 3d positions themselves. """
        numCameras = self.cameraArray.shape[0]
        points_3d = self.points3D

        x0 = self.cameraArray.ravel()
        A = self.jac_sparsity_justcam(self, numCameras, self.cameraIndices)
        res = least_squares(self.fun_justcam, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-8, method='trf', jac='3-point',
                            args=(points_3d, numCameras, self.cameraIndices, self.point2DIndices, self.points2D, self.pointWeights))
        self.cameraArray = res.x.reshape((n_cameras, nCamParams))

        return res

    def bundleAdjust_just_extrinsics(self):
        """ Returns the optimized camera extrinsics given known 3d positions,
        without adjusting the 3d positions themselves and holding the instrinsic camera params fixed. """
        n_extrinsics = 6
        numCameras = self.cameraArray.shape[0]
        points_3d = self.points3D
        cam_intrinsics = self.cameraArray[:, n_extrinsics:]

        x0 = self.cameraArray[:, :n_extrinsics].ravel()
        A = self.jac_sparsity_just_extrinsics(self, numCameras, self.cameraIndices)
        res = least_squares(self.fun_just_extrinsics, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-8, method='trf', jac='3-point',
                            args=(points_3d, numCameras, self.cameraIndices, self.point2DIndices, self.points2D, self.pointWeights))
        
        cam_extrinsics = res.x.reshape((n_cameras, n_extrinsics))
        self.cameraArray = np.column_stack([cam_extrinsics, cam_intrinsics])

        return res