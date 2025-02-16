from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import dataloader as dt

from models.test_lightning import Model

train_loader, test_loader = dt.get_dataloaders('/media/data/iter3/train', '/media/data/iter3/target', 64, 8)

# Set up logging and checkpointing
logger = TensorBoardLogger("logs", name="val_test2")
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",  # Metric to monitor
    save_top_k=1,        # Save only the best model
    mode="min"           # Minimize the monitored metric (e.g., val_loss)
)

# Train the model
trainer = Trainer(
    logger=logger,
    callbacks=[checkpoint_callback],
    max_epochs=1
)
model = Model()
trainer.fit(model, train_loader, test_loader)