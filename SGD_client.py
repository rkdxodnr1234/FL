import sys
import json
import torch
import flwr as fl
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from simpleCNN import DeepCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔹 사용자 정의 FEMNIST 데이터셋 클래스
class FEMNISTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# 🔹 FEMNIST 데이터 로드 함수
def load_femnist_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    user_key = list(data['user_data'].keys())[0]
    images = torch.tensor(data['user_data'][user_key]['x']).reshape(-1, 1, 28, 28).float()
    labels = torch.tensor(data['user_data'][user_key]['y']).long()
    return images, labels

# 🔹 Flower 클라이언트 클래스 (FedSGD 구현)
class FLClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloader):
        self.model = model.to(device) 
        self.trainloader = trainloader
        self.testloader = testloader
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self):
        """모델 파라미터를 NumPy 배열로 반환"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """NumPy 배열로 전달받은 파라미터를 모델에 설정"""
        state_dict = self.model.state_dict()
        for key, val in zip(state_dict.keys(), parameters):
            state_dict[key] = torch.tensor(val)
        self.model.load_state_dict(state_dict)

    def train(self):
        """FedSGD: 한 번의 SGD 업데이트만 수행"""
        self.model.train()
        
        # 첫 번째 미니배치에서 한 번의 SGD 업데이트만 수행
        for images, labels in self.trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer = optim.SGD(self.model.parameters(), lr=0.05)  # 학습률 설정
            optimizer.zero_grad()
            output = self.model(images)
            loss = self.criterion(output, labels)
            loss.backward()
            optimizer.step()
            break  # 한 번의 업데이트만 수행
        
        return self.get_parameters()

    def evaluate_model(self):
        """모델 평가: 정확도와 손실 계산"""
        self.model.eval()
        correct, total, loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in self.testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                loss += self.criterion(outputs, labels).item()
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
        
        acc = correct / total if total > 0 else 0.0
        print(f"[CLIENT EVAL] Accuracy: {correct}/{total} = {acc:.4f}")
        
        return loss / len(self.testloader), acc

    def fit(self, parameters, config):
        """FedSGD 학습: 한 번의 SGD 업데이트 수행 후 파라미터 반환"""
        self.set_parameters(parameters)  # 서버에서 전달받은 파라미터 설정
        updated_parameters = self.train()  # 한 번의 SGD 업데이트 수행
        
        return updated_parameters, len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        """모델 평가: 손실과 정확도 반환"""
        self.set_parameters(parameters)  # 서버에서 전달받은 파라미터 설정
        loss, accuracy = self.evaluate_model()  # 모델 평가
        
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}

# 🔹 실행 시작
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❗ 클라이언트 실행 시 학습 데이터 경로를 인자로 전달해야 합니다.")
        sys.exit(1)

    # 학습 데이터 로드 및 DataLoader 생성
    train_path = sys.argv[1]
    train_images, train_labels = load_femnist_data(train_path)
    trainloader = DataLoader(FEMNISTDataset(train_images, train_labels), batch_size=32, shuffle=True)

    # 테스트 데이터 로드 및 DataLoader 생성 (테스트 데이터가 없을 경우 학습 데이터를 사용)
    test_path = train_path.replace("train", "test").replace("_train_", "_test_")
    try:
        test_images, test_labels = load_femnist_data(test_path)
    except FileNotFoundError:
        print(f"❗ 테스트 데이터 파일을 찾을 수 없습니다: {test_path}")
        test_images, test_labels = train_images, train_labels  # fallback to training data
    
    testloader = DataLoader(FEMNISTDataset(test_images, test_labels), batch_size=32, shuffle=False)

    # 모델 및 클라이언트 초기화
    model = DeepCNN()
    client = FLClient(model=model, trainloader=trainloader, testloader=testloader)

    # Flower 클라이언트 시작 (FedSGD 방식으로 동작)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
