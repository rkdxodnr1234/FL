import os
import json
import flwr as fl
import numpy as np
import pickle
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from simpleCNN import DeepCNN  # DeepCNN 모델 사용

# 🔹 결과 저장 경로 (공통)
SAVE_DIR = r"C:/Users/ahsld/Desktop/연합학습 시뮬/학술/sgd 결과"
os.makedirs(SAVE_DIR, exist_ok=True)

# 🔹 정확도 및 손실 저장 리스트
accuracy_history = []
loss_history = []

# 🔹 메트릭 저장 함수
def save_metrics(round_num, accuracy, loss):
    with open(os.path.join(SAVE_DIR, f"fedsgd_round_{round_num}_metrics.json"), 'w') as f:
        json.dump({"accuracy": accuracy, "loss": loss}, f)

# 🔹 체크포인트 저장 (pickle로 수정)
def save_checkpoint(server_round, parameters):
    checkpoint_path = os.path.join(SAVE_DIR, f"fedsgd_round_{server_round}_checkpoint.pkl")
    aggregated_ndarrays = parameters_to_ndarrays(parameters)
    with open(checkpoint_path, "wb") as f:
        pickle.dump(aggregated_ndarrays, f)
    print(f"✅ 모델 체크포인트 저장 완료 (FedSGD): {checkpoint_path}")

# 🔹 커스텀 전략 정의 (FedSGD 기반)
class CustomFedSGD(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        aggregated_result = super().aggregate_fit(server_round, results, failures)
        if aggregated_result is not None:
            parameters_aggregated = aggregated_result[0]
            save_checkpoint(server_round, parameters_aggregated)
        return aggregated_result

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None

        accuracies = [res.metrics["accuracy"] * res.num_examples for _, res in results if "accuracy" in res.metrics]
        examples = [res.num_examples for _, res in results]
        avg_accuracy = sum(accuracies) / sum(examples) if examples else 0.0
        avg_loss = float(np.mean([res.loss for _, res in results])) if results else 0.0

        print(f"[ROUND {server_round}] Aggregated Accuracy: {avg_accuracy:.4f}, Aggregated Loss: {avg_loss:.4f}")
        save_metrics(round_num=server_round, accuracy=avg_accuracy, loss=avg_loss)

        return super().aggregate_evaluate(server_round=server_round, results=results, failures=failures)

# 🔹 초기 모델 설정
initial_model = DeepCNN()
initial_parameters = ndarrays_to_parameters(
    [val.cpu().numpy() for _, val in initial_model.state_dict().items()]
)

# 🔹 FedSGD 전략 설정
strategy = CustomFedSGD(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=20,
    min_evaluate_clients=20,
    min_available_clients=20,
    initial_parameters=initial_parameters,
)

# 🔹 서버 실행
if __name__ == "__main__":
    history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=1000),
        strategy=strategy,
    )

    # 저장된 메트릭 불러오기 (그래프는 따로)
    for round_num in range(1, 1001):
        try:
            with open(os.path.join(SAVE_DIR, f"fedsgd_round_{round_num}_metrics.json"), 'r') as f:
                metrics = json.load(f)
                accuracy_history.append(metrics["accuracy"])
                loss_history.append(metrics["loss"])
        except FileNotFoundError:
            break
