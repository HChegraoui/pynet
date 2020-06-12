from pprint import pprint

from torch.optim import lr_scheduler

from pynet import NetParameters
from pynet.datasets import DataManager, fetch_metastasis
from pynet.interfaces import UNetSegmenter
from pynet.losses import SoftDiceLoss
from pynet.models import UNet
from pynet.plotting import plot_data
from pynet.plotting import plot_history
from pynet.utils import get_named_layers
from pynet.utils import setup_logging

setup_logging(level="debug")

model = UNet(num_classes=4, input_shape=(128, 128, 128))
layers = get_named_layers(model)
pprint(layers)
data = fetch_metastasis()
manager = DataManager(
    input_path=data.input_path,
    metadata_path=data.metadata_path,
    output_path=data.output_path,
    projection_labels=None,
    number_of_folds=10,
    batch_size=1,
    stratify_label="grade",
    sampler="random",
    add_input=True,
    test_size=0.1,
    pin_memory=True)
dataset = manager["test"][:1]
print(dataset.inputs.shape, dataset.outputs.shape)
plot_data(dataset.inputs, channel=1, nb_samples=5)
plot_data(dataset.outputs, channel=1, nb_samples=5)
my_loss = SoftDiceLoss()
outdir = "/neurospin/radiomics/workspace_biomede/tests/unet"
net_params = NetParameters(
    input_shape=(150, 181, 137),
    in_channels=4,
    num_classes=4,
    activation="relu",
    normalization="group_normalization",
    mode="trilinear",
    with_vae=True)
unet = UNetSegmenter(
    net_params,
    optimizer_name="Adam",
    learning_rate=1e-4,
    weight_decay=1e-5,
    loss=my_loss,
    use_cuda=True)
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer=unet.optimizer,
    mode="min",
    factor=0.5,
    patience=5)
train_history, valid_history = unet.training(
    manager=manager,
    nb_epochs=100,
    checkpointdir=outdir,
    # fold_index=0,
    scheduler=scheduler,
    with_validation=True)
print(train_history)
print(valid_history)
plot_history(train_history)

y_pred, X, y_true, loss, values = unet.testing(
    manager=manager,
    with_logit=True,
    predict=True)
print(y_pred.shape, X.shape, y_true.shape)
