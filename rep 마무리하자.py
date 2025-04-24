import os
import json
import pickle
import flwr as fl
import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from simpleCNN import DeepCNN

# 🔹 저장 경로 설정
SAVE_DIR = r"C:\Users\ahsld\Desktop\연합학습 시뮬\전체 지표 출력\FedRep 결과2"
os.makedirs(SAVE_DIR, exist_ok=True)

# 🔹 메트릭 저장 함수
def save_metrics(round_num, accuracy, loss):
    with open(os.path.join(SAVE_DIR, f"round_{round_num}_metrics.json"), "w") as f:
        json.dump({"accuracy": accuracy, "loss": loss}, f)

# 🔹 모델 체크포인트 저장 함수
def save_checkpoint(round_num, parameters):
    path = os.path.join(SAVE_DIR, f"round_{round_num}_checkpoint.pkl")
    weights = parameters_to_ndarrays(parameters)
    with open(path, "wb") as f:
        pickle.dump(weights, f)
    print(f"✅ 모델 체크포인트 저장 완료: {path}")

# 🔹 FedRep 전략 정의
class FedRepStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None

        # feature extractor 파라미터만 평균화
        all_weights = [parameters_to_ndarrays(res.parameters) for _, res in results]
        avg_weights = [np.mean([w[i] for w in all_weights], axis=0) for i in range(len(all_weights[0]))]
        aggregated_params = ndarrays_to_parameters(avg_weights)

        save_checkpoint(server_round, aggregated_params)
        return aggregated_params, {}

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None
        accuracies = [res.metrics["accuracy"] * res.num_examples for _, res in results]
        losses = [res.loss for _, res in results]
        examples = [res.num_examples for _, res in results]

        avg_acc = sum(accuracies) / sum(examples) if examples else 0.0
        avg_loss = np.mean(losses) if losses else 0.0

        save_metrics(server_round, avg_acc, avg_loss)
        print(f"[ROUND {server_round}] Accuracy: {avg_acc:.4f}, Loss: {avg_loss:.4f}")
        return avg_loss, {"accuracy": avg_acc}

# 🔹 초기 모델 생성
initial_model = DeepCNN()
initial_params = ndarrays_to_parameters(
    [val.cpu().numpy() for _, val in initial_model.feature_extractor.state_dict().items()]
)

# 🔹 전략 설정
strategy = FedRepStrategy(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=20,
    min_evaluate_clients=20,
    min_available_clients=20,
    initial_parameters=initial_params,
)

# 🔹 서버 실행
if __name__ == "__main__":
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=100),
        strategy=strategy,
    )
