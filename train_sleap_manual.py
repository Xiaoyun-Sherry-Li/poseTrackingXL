cfg = sleap.load_config("Z:/Sherry/poseTrackingXL/training_files/SLP/models/250430_222637.single_instance.n=1684")

cfg.outputs.run_name = "Z:/Sherry/poseTrackingXL/training_files/SLP/models/transfer"

trainer.keras_model.load_weights("models/baseline_model.topdown/best_model.h5")

#%%
import sleap
sleap.versions()
sleap.system_summary()

#%%
TRAINING_SLP_FILE = "Z:/Sherry/poseTrackingXL/training_files/SLP/043025_com_net.slp"
cfg = sleap.load_config("Z:/Sherry/poseTrackingXL/training_files/SLP/models/comNet250430_222637.single_instance.n=1684/training_config.json")

#%%
trainer = sleap.nn.training.Trainer.from_config(cfg)
trainer.train()

#%%
trainer.config.optimization.epochs = 3
trainer.train()
# Load config.
cfg = sleap.load_config("models/baseline_model.topdown")
# cfg.outputs.run_name = "new_folder"  # Set the run_name to a new value if you want the model to be saved to a different folder.
# Create and initialize the trainer.
trainer = sleap.nn.training.Trainer.from_config(cfg)
trainer.setup()
# Replace the randomly initialized weights with the saved weights.
trainer.keras_model.load_weights("models/baseline_model.topdown/best_model.h5")