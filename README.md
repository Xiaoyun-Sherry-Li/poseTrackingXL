# bird_pose_tracking
3D pose tracking for chickadees in a caching arena, from Selmaan Chettih's **supervisely_dpk_training** repo.

**Utils**
- Bundle adjustment for calibrating views from multiple cameras (pySBA).
- Markerless pose tracking (SLEAP).

**Workflow**
- Identifies coarse 2D keypoints from raw video frames.
- Triangulates these keypoints across multiple cameras for 3D location info.
- Uses the bird's location to resample a cropped image around the bird (fixed pixel size).
- Identifies fine-res 2D postural keypoints.
- Triangulates with other views for 3D postural positions. 

### Step 0: Calibrate cameras
For 3D pose tracking, we need to use bundle adjustment to reconcile views across multiple cameras. The goal is to obtain an optimized camera array, which defines how each camera view translates into 3D world coordinates. We’ll do this using a laser calibration method developed by Selmaan Chettih with Rob Johnson and Jinyao (Jane) Yan at Janelia.

**Required repositories**
- `Label3D` from [Diego Aldorando](https://github.com/diegoaldarondo/Label3D)
- `il_rig_control` [from me](https://github.com/isabellow/il_rig_control) (based on code by Selmaan)
- most functions are in the `camera_calibration` folder in this repo
- `pySBA` (included in this repo) is the bundle adjustment code [from Jahdiel](https://github.com/jahdiel/pySBA)

**Set-up and collect the calibration videos**
- Use `init_cam_array.ipynb` to create an initial estimate of the camera arrays. This notebook has basic notes about the camera array and links to other resources.
- Define your known points
  - This is a set of points with known real-world coordinates, which you can then translate into whatever normalized coordinate frame you like.
  - You can use `init_world_coord_feeders.ipynb` to set the known point locations
  - To use this function, you will need an arena model image, which has the location of key landmarks (e.g., feeders, cache sites).
- Use `bird_pose_tracking/camera_calibration/collect_calibration_video` bonsai script to collect calibration videos
  - This is a video of a laser pointer moving around the arena. Be sure to cover most of the FOV of each camera.
  - This results in many frames, each with one point seen by multiple cameras
- Collect a few frames video of the empty arena with known points marked

**Preprocess the videos to extract centroid locations**
- Depending on your video reader, either convert the videos to images (`save_calibration_images.ipynb`) or directly read in the frames as images
- Use `extract_preprocess_centroids.ipynb` to extract and preprocess the laser pointer centroids.
- This results in a set of 2D points for each camera, indexed by which frame they correspond to. We can now use bundle adjustment to determine the 3D location of these points.

**Calibrate the cameras to get an optimized array**
- Use `centroid_processing_clean.ipynb` to calibrate the cameras.
- Load an existing optimized array (either initial estimate or old array).
- Use the 2D laser points to optimize over the 3D point locations and the camera array parameters.
  - This should result in very low reprojection error (99th percentile around 0.7 pixels or less, bulk of the distribution near 0.1 - 0.2 pixels).
- Use `arena_coordinate_transform_2.m` to use `Label3D` to label the known points in the 2D camera views, triangulating their 3D locations using the newly optimized array.
- Update the camera extrinsics to map the current 3D locations of the known points onto your defined, normalized coordinate system.
- Update the 3D points locations and check that the reprojection error has not changed.

### Step 1: Train models

### Step 2: Infer 3D posture
