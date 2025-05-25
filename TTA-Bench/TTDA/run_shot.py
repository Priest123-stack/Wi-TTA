
import torch
import numpy as np
from main import CNN
from main import setup_shot
from torch.utils.data import DataLoader, TensorDataset

x_train = np.load('E:/wifi感知/5300-3_npy/x_train.npy')
x_test = np.load('E:/wifi感知/5300-3_npy/x_test.npy')
y_train = np.load('E:/wifi感知/5300-3_npy/y_train.npy')
y_test = np.load('E:/wifi感知/5300-3_npy/y_test.npy')

x_comb = np.concatenate([x_train, x_test], axis=0)
y_comb = np.concatenate([y_train, y_test], axis=0)

x_comb = torch.tensor(x_comb.reshape(len(x_comb), 1, 2000, 30), dtype=torch.float32)
y_comb = torch.tensor(y_comb, dtype=torch.long)
combined_dataset = TensorDataset(x_comb, y_comb)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'D:/model/5300-3.pth'

def test_time_adaptation_local(net, test_loader, device):
    net.train()
    correct = total = 0

    for i, (x_test, y_test) in enumerate(test_loader):
        x_test, y_test = x_test.float().to(device), y_test.to(device)
        total += y_test.size(0)
        outputs = net(x_test)
        pred = outputs.argmax(1)
        correct += (pred == y_test).sum().item()

    acc = correct / total
    return acc

def run_multiple_shot_evaluations(n_runs=3, batch_size=64):
    accs = []
    for seed in [2019, 2020, 2021][:n_runs]:
        print(f"\n=== Run with seed {seed} ===")
        torch.manual_seed(seed)
        np.random.seed(seed)

        loader = DataLoader(dataset=combined_dataset, batch_size=batch_size, shuffle=True)

        model = CNN()
        model.load_state_dict(torch.load(model_path, map_location=device,weights_only=True))
        model.to(device)
        model.forward_features = lambda x: model.layer4(model.layer3(model.layer2(model.layer1(x)))).view(x.size(0), -1)

        shot_model = setup_shot(model)
        acc = test_time_adaptation_local(shot_model, loader, device)
        accs.append(acc)
        print(f"Accuracy: {acc:.4f}")

    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    print(f"\nAverage Accuracy over {n_runs} runs: {mean_acc:.4f} ± {std_acc:.4f}")

if __name__ == "__main__":
    run_multiple_shot_evaluations()
