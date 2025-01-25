from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Define your model
class MyModel(LightningModule):
    def __init__(self):
        super().__init__()
        # Your model, loss, optimizer, etc.

# Set up logging and checkpointing
logger = TensorBoardLogger("logs", name="my_model")
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",  # Metric to monitor
    save_top_k=1,        # Save only the best model
    mode="min"           # Minimize the monitored metric (e.g., val_loss)
)

# Train the model
trainer = Trainer(
    logger=logger,
    callbacks=[checkpoint_callback],
    max_epochs=10
)
model = MyModel()
trainer.fit(model)