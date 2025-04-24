import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
from torch.utils.data import Dataset, DataLoader
from simpleCNN import DeepCNN  # CNN 모델 정의
from typing import Tuple

# 장치 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔹 FEMNIST 데이터셋 로딩
def load_femnist_data(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    with open(path, 'r') as f:
        data = json.load(f)
    user_key = list(data['user_data'].keys())[0]
    images = torch.tensor(data['user_data'][user_key]['x']).reshape(-1, 1, 28, 28).float()
    labels = torch.tensor(data['user_data'][user_key]['y']).long()
    return images, labels

# 🔹 Dataset 클래스 정의
class FEMNISTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# 🔹 FedRep 클라이언트 정의
class FLClient(fl.client.NumPyClient):
    def __init__(self, trainloader, testloader):
        self.model = DeepCNN().to(DEVICE)
        self.trainloader = trainloader
        self.testloader = testloader
        self.criterion = nn.CrossEntropyLoss()
        self.lr = 0.01
        self.local_epochs = 100

    def get_parameters(self):
        # feature extractor만 전달
        return [val.cpu().numpy() for _, val in self.model.feature_extractor.state_dict().items()]

    def set_parameters(self, parameters):
        # feature extractor만 업데이트
        state_dict = self.model.feature_extractor.state_dict()
        for k, val in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(val)
        self.model.feature_extractor.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # 🔹 Step 1: 공통 표현 학습 (feature extractor 고정)
        self.model.head.requires_grad_(True)
        self.model.feature_extractor.requires_grad_(True)
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        self.model.train()
        for _ in range(self.local_epochs):
            for x, y in self.trainloader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                out = self.model(x)
                loss = self.criterion(out, y)
                loss.backward()
                optimizer.step()

        # 🔹 Step 2: 개인화된 head만 추가로 학습
        self.model.feature_extractor.requires_grad_(False)
        self.model.head.requires_grad_(True)
        optimizer = optim.SGD(self.model.head.parameters(), lr=self.lr)
        for _ in range(self.local_epochs):
            for x, y in self.trainloader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                out = self.model(x)
                loss = self.criterion(out, y)
                loss.backward()
                optimizer.step()

        return self.get_parameters(), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct, total, loss_total = 0, 0, 0.0
        with torch.no_grad():
            for x, y in self.testloader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss_total += loss.item()
                correct += (outputs.argmax(1) == y).sum().item()
                total += y.size(0)
        acc = correct / total if total > 0 else 0.0
        return loss_total / len(self.testloader), total, {"accuracy": acc}

# 🔹 실행부
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python FedRep_client.py <데이터경로>")
        sys.exit(1)

    train_path = sys.argv[1]
    test_path = train_path.replace("train", "test").replace("_train_", "_test_")

    train_x, train_y = load_femnist_data(train_path)
    test_x, test_y = load_femnist_data(test_path)

    trainloader = DataLoader(FEMNISTDataset(train_x, train_y), batch_size=32, shuffle=True)
    testloader = DataLoader(FEMNISTDataset(test_x, test_y), batch_size=32, shuffle=False)

    client = FLClient(trainloader, testloader)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
