import os
import json
import pickle
import flwr as fl
import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from simpleCNN import DeepCNN

# 🔹 저장 경로 설정
SAVE_DIR = r"C:/Users/ahsld/Desktop/연합학습 시뮬/학술/avgnew"
os.makedirs(SAVE_DIR, exist_ok=True)

# 🔹 정확도 및 손실 저장 리스트
accuracy_history = []
loss_history = []

# 🔹 메트릭 저장 함수
def save_metrics(round_num, accuracy, loss):
    with open(os.path.join(SAVE_DIR, f"round_{round_num}_metrics.json"), 'w') as f:
        json.dump({"accuracy": accuracy, "loss": loss}, f)

# 🔹 수렴 속도 계산 함수 (정확도 70% 기준)
def find_convergence_round(acc_list, threshold=0.70):
    for i, acc in enumerate(acc_list, start=1):
        if acc >= threshold:
            return i
    return None

# 🔹 최종 요약 저장 함수
def save_summary_metrics():
    summary = {
        "final_accuracy": round(accuracy_history[-1], 4) if accuracy_history else None,
        "final_loss": round(loss_history[-1], 4) if loss_history else None,
        "convergence_round_70%": find_convergence_round(accuracy_history, threshold=0.70),
        "num_rounds": len(accuracy_history),
        "communication_efficiency": round(sum(loss_history), 4)
    }
    with open(os.path.join(SAVE_DIR, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
    print("✅ 메트릭 요약 저장 완료:", summary)

# 🔹 모델 체크포인트 저장 함수
def save_checkpoint(server_round, parameters):
    """모델 파라미터를 체크포인트로 저장 (pickle 방식)"""
    checkpoint_path = os.path.join(SAVE_DIR, f"round_{server_round}_checkpoint.pkl")
    aggregated_ndarrays = parameters_to_ndarrays(parameters)

    with open(checkpoint_path, "wb") as f:
        pickle.dump(aggregated_ndarrays, f)

    print(f"✅ 모델 체크포인트 저장 완료 (pickle): {checkpoint_path}")

# 🔹 사용자 정의 FedAvg 전략
class CustomFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        aggregated_result = super().aggregate_fit(server_round, results, failures)
        if aggregated_result is not None:
            parameters_aggregated = aggregated_result[0]
            save_checkpoint(server_round, parameters_aggregated)
        return aggregated_result

    def aggregate_evaluate(self, server_round, results, failures):
        accuracies = [res.metrics["accuracy"] for _, res in results if "accuracy" in res.metrics]
        losses = [res.loss for _, res in results]
        avg_accuracy = float(np.mean(accuracies)) if accuracies else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0

        print(f"[ROUND {server_round}] Aggregated Accuracy: {avg_accuracy:.4f}, Aggregated Loss: {avg_loss:.4f}")
        save_metrics(round_num=server_round, accuracy=avg_accuracy, loss=avg_loss)

        return super().aggregate_evaluate(server_round=server_round, results=results, failures=failures)

# 🔹 초기 모델 파라미터 설정
initial_model = DeepCNN()
initial_parameters = ndarrays_to_parameters(
    [val.cpu().numpy() for _, val in initial_model.state_dict().items()]
)

# 🔹 전략 정의
strategy = CustomFedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=5,
    min_evaluate_clients=5,
    min_available_clients=5,
    initial_parameters=initial_parameters,
)

# 🔹 서버 실행
if __name__ == "__main__":
    history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=100),
        strategy=strategy,
    )

    # 각 라운드별 저장된 메트릭 불러오기
    for round_num in range(1, 101):
        try:
            with open(os.path.join(SAVE_DIR, f"round_{round_num}_metrics.json"), 'r') as f:
                metrics = json.load(f)
                accuracy_history.append(metrics["accuracy"])
                loss_history.append(metrics["loss"])
        except FileNotFoundError:
            break

    save_summary_metrics()
