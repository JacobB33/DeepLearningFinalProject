from torch.utils.data import DataLoader
from encoder import  BrainScanEmbedder
from training.data import  get_train_dataset
from training.configs import ModelConfig

data = get_train_dataset('/home/jacob/projects/DeepLearningFinalProject/data/processed_data')
cfg = ModelConfig([8, 16, 32, 64, 77])
model = BrainScanEmbedder(cfg)
for input, target in DataLoader(data, batch_size=23):
    model = model.forward(input, target)