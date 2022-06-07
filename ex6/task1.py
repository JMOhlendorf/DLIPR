from comet_ml import Experiment
import numpy as np
import dlipr
from tensorflow import keras
inc = keras.applications.inception_v3


# Set up YOUR experiment - login to comet, create new project (for new exercise)
# and copy the statet command
# or just change the name of the workspace, and the API (you can find it in the settings)
experiment = Experiment(api_key="EnterYourAPIKey",
                        project_name="exercise6", workspace="EnterGroupWorkspaceHere")

# ------------------------------------------------------------
# Apply a trained ImageNet classification network to classify new images.
# See https://keras.io/applications/ for further instructions.
# ------------------------------------------------------------

# Note: Keras downloads the pretrained network weights to ~/.keras/models.
# To save space in your home folder you can use the /net/scratch/deeplearning/keras-models folder.
# Simply open the terminal and copy/paste:
# ln -s /net/scratch/deeplearning/keras-models ~/.keras/models
# If you get an error "cannot overwrite directory", remove the existing .keras/models folder first.
# Alternatively, you can set up the model with "weights=None" and then use model.load_weights('/net/scratch/deeplearning/keras-models/...')

# Example: Inception-v3
model = inc.InceptionV3(weights='imagenet')
