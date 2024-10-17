import cv2
import numpy as np

def camera_matrix(K, R, t):
    """Derive the camera matrix.
    Derive the camera matrix from the camera intrinsic matrix (K),
    and the extrinsic rotation matric (R), and extrinsic
    translation vector (t).
    Note that this uses the matlab convention, such that
    M = [R;t] * K
    """
    return np.concatenate((R, t), axis=0) @ K

def unDistortPoints(pts, intrinsicMatrix, radialDistortion,
    tangentDistortion=np.array([0,0])):
    """Remove lens distortion from the input points.
    Input is size (M,2), where M is the number of points
    """
    dcoef = radialDistortion.ravel()[:2].tolist() + tangentDistortion.ravel().tolist()

    if len(radialDistortion.ravel()) == 3:
        dcoef = dcoef + [radialDistortion.ravel()[-1]]
    else:
        dcoef = dcoef + [0]

    pts_u = cv2.undistortPoints(
        np.reshape(pts, (-1, 1, 2)).astype("float32"),
        intrinsicMatrix.T,
        np.array(dcoef),
        P=intrinsicMatrix.T,
    )

    pts_u = np.reshape(pts_u, (-1, 2))

    return pts_u

def triangulate(pts1, pts2, cam1, cam2):
    """Return triangulated 3- coordinates.
    Following Matlab convention, given lists of matching points, and their
    respective camera matrices, returns the triangulated 3- coordinates.
    pts1 and pts2 must be Mx2, where M is the number of points with
    (x,y) positions. M 3-D points will be returned after triangulation
    """
    pts1 = pts1.T
    pts2 = pts2.T

    cam1 = cam1.T
    cam2 = cam2.T

    out_3d = np.zeros((3, pts1.shape[1]))

    for i in range(out_3d.shape[1]):
        if ~np.isnan(pts1[0, i]):
            pt1 = pts1[:, i : i + 1]
            pt2 = pts2[:, i : i + 1]

            A = np.zeros((4, 4))
            A[0:2, :] = pt1 @ cam1[2:3, :] - cam1[0:2, :]
            A[2:, :] = pt2 @ cam2[2:3, :] - cam2[0:2, :]

            u, s, vh = np.linalg.svd(A)
            v = vh.T

            X = v[:, -1]
            X = X / X[-1]

            out_3d[:, i] = X[0:3].T
        else:
            out_3d[:, i] = np.nan

    return out_3d

def triangulate_single_pt(pts, cameraMats):
    """Return triangulated 3- coordinates.
    Following Matlab convention, given lists of matching points, and their
    respective camera matrices, returns the triangulated 3- coordinates.
    pts must be Nx2, where N is the number of views used for triangulation
    camMats must be a list w/ length N where the Nth item is the 4x3 camera matrix for view n.
    """

    numViews = len(cameraMats)
    A = np.zeros((numViews*2,4))
    for idx in range(numViews):
        thisPt = pts[idx]
        thisMat = cameraMats[idx].T
        A[2*idx:2*idx+2] = np.outer(thisPt, thisMat[2]) - thisMat[0:2]

    u,s,v = np.linalg.svd(A)
    return v[-1,0:-1]/v[-1,-1]

def reproject_single_pt(pt3d, cameraMats):
    pt2d = []
    for idx in range(len(cameraMats)):
        pt2dh = np.dot(np.hstack((pt3d,1)), cameraMats[idx])
        pt2d.append(pt2dh[:2] / pt2dh[2])
    return np.stack(pt2d)

def triangulate_best_3set(pts, cameraMats, nCams=6):
    """Wrapper around triangulate_single_pt. Takes in a nViewsx2 matrix of 2d pts
    and a list of 4x3 camera matrices cameraMats, triangulates all combinations of 3 cameras and
    returns the location and reprojection error for the best one
        """
    if pts.shape[0] != len(cameraMats):
        print('Number of points does not equal number of camera matrices')
        raise ValueError
    best_err = np.Inf
    for iCam in range(nCams):
        for jCam in range(iCam+1, nCams):
            for kCam in range(jCam+1, nCams):
                thesePts = pts[[iCam, jCam, kCam]]
                theseMats = [cameraMats[i] for i in [iCam, jCam, kCam]]
                p3d = triangulate_single_pt(thesePts, theseMats)
                err = np.sqrt(np.sum((reproject_single_pt(p3d, theseMats) - thesePts) ** 2, axis=1)).mean()
                if err < best_err:
                    best_err = err
                    best_p3d = p3d
    return best_p3d, best_err

def triangulate_3sets(pts, cameraMats, nSelections=5, nCams=6):
    """Wrapper around triangulate_single_pt. Takes in a nViewsx2 matrix of 2d pts
    and a list of 4x3 camera matrices cameraMats, triangulates all combinations of 3 cameras,
    finds the top nSelections among them, and returns their average or median position
    and reprojection error"""
    if pts.shape[0] != len(cameraMats):
        print('Number of points does not equal number of camera matrices')
        raise ValueError

    allPts = np.zeros((np.math.factorial(nCams)//np.math.factorial(3)**2, 3))
    allErr = np.zeros((np.math.factorial(nCams)//np.math.factorial(3)**2))
    nCombo = -1
    for iCam in range(nCams):
        for jCam in range(iCam+1, nCams):
            for kCam in range(jCam+1, nCams):
                nCombo += 1
                thesePts = pts[[iCam, jCam, kCam]]
                theseMats = [cameraMats[i] for i in [iCam, jCam, kCam]]
                allPts[nCombo] = triangulate_single_pt(thesePts, theseMats)
                allErr[nCombo] = np.sqrt(np.sum((reproject_single_pt(allPts[nCombo], theseMats) - thesePts) ** 2, axis=1)).mean()

    best_idx = np.argpartition(allErr, nSelections)[:nSelections]
    best_p3d = np.mean(allPts[best_idx],axis=0)
    best_err = np.mean(allErr[best_idx])

    return best_p3d, best_err

def triangulate_confThresh_lowestErr(pts, cameraMats, conf, kSelect=4):
    """Wrapper around triangulate_single_pt. Takes in a nViewsx2 matrix of 2d pts,
    a list of 4x3 camera matrices cameraMats, and an nViews vector of confidence scores,
    selects the k highest confidence values, triangulates all combinations of 3 cameras,
    selects the triangulation with minimum reprojection error, and returns the point and
    reprojection error"""
    nCams = pts.shape[0]
    if nCams != len(cameraMats):
        print('Number of points does not equal number of camera matrices')
        raise ValueError

    # Get index of top k confidence scores
    best_idx = np.argpartition(conf, -kSelect)[-kSelect:]
    # Get triangulations, confidence, and reprojection errors for each set of 3
    best_err = np.Inf
    for iCam in range(kSelect):
        for jCam in range(iCam+1, kSelect):
            for kCam in range(jCam+1, kSelect):
                camInd = best_idx[[iCam, jCam, kCam]]
                thesePts = pts[camInd]
                theseMats = [cameraMats[i] for i in camInd]
                p3d = triangulate_single_pt(thesePts, theseMats)
                err = np.sqrt(np.sum((reproject_single_pt(p3d, theseMats) - thesePts) ** 2, axis=1)).mean()
                if err < best_err:
                    best_err = err
                    best_p3d = p3d
                    best_conf = conf[camInd].mean()
    return best_p3d, best_err, best_conf

def triangulate_confThresh_medPair(pts, cameraMats, conf, kSelect=3):
    """Wrapper around triangulate_single_pt. Takes in a nViewsx2 matrix of 2d pts,
    a list of 4x3 camera matrices cameraMats, and an nViews vector of confidence scores,
    selects the k highest confidence values, triangulates all combinations of 2 cameras,
    and returns the median point and reprojection error"""
    nCams = pts.shape[0]
    if nCams != len(cameraMats):
        print('Number of points does not equal number of camera matrices')
        raise ValueError

    # Get index of top k confidence scores
    best_idx = np.argpartition(conf, -kSelect)[-kSelect:]
    mean_conf = conf[best_idx].mean()
    # Get triangulations, confidence, and reprojection errors for each pair
    nCombos = np.math.factorial(kSelect) // (np.math.factorial(2) * np.math.factorial(kSelect-2))
    allPts = np.zeros((nCombos, 3))
    nCombo = -1
    for iCam in range(kSelect):
        for jCam in range(iCam+1, kSelect):
            nCombo += 1
            camInd = best_idx[[iCam, jCam]]
            thesePts = pts[camInd]
            theseMats = [cameraMats[i] for i in camInd]
            allPts[nCombo] = triangulate_single_pt(thesePts, theseMats)
    medPt = np.median(allPts, axis=0)
    meanErr = np.sqrt(np.sum((reproject_single_pt(medPt, [cameraMats[i] for i in best_idx]) -
                              pts[best_idx]) ** 2, axis=1)).mean()

    return medPt, meanErr, mean_conf
