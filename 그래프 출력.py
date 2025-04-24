import os
import json
import matplotlib.pyplot as plt

# 오류 해결: 저장 경로가 없을 경우 생성
os.makedirs(BASE_DIR, exist_ok=True)

# 정확도 그래프 다시 시도
plt.figure(figsize=(10, 6))
for name in results:
    plt.plot(results[name]["accuracy"], label=name, color=colors[name])
plt.title("Accuracy per Round")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "graph_accuracy_combined.png"))
plt.close()

# 손실 그래프 다시 시도
plt.figure(figsize=(10, 6))
for name in results:
    plt.plot(results[name]["loss"], label=name, color=colors[name])
plt.title("Loss per Round")
plt.xlabel("Round")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "graph_loss_combined.png"))
plt.close()

# 통신 효율성 막대 그래프
plt.figure(figsize=(8, 5))
names = list(results.keys())
comm_values = [results[name]["comm_rounds"] for name in names]
plt.bar(names, comm_values, color=[colors[name] for name in names])
plt.title("Communication Efficiency (Total Rounds)")
plt.ylabel("Rounds")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "graph_comm_efficiency_combined.png"))
plt.close()

# 수렴 속도 막대 그래프
plt.figure(figsize=(8, 5))
conv_values = [results[name]["converged_round"] for name in names]
plt.bar(names, conv_values, color=[colors[name] for name in names])
plt.title("Convergence Speed (Rounds to 70% Accuracy)")
plt.ylabel("Rounds")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "graph_convergence_speed_combined.png"))
plt.close()
