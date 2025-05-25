import subprocess
import re
import statistics


def run_main_and_capture_accuracy():
    # 运行 main.py 并捕获输出
    result = subprocess.run(
        ["python", "main.py"],
        # ["python", "3C-GAN.py"],
        # ["python", "GeOS.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    output = result.stdout

    # 使用正则表达式匹配准确度（匹配 test_acc 后的数字，如 test_acc0.755）
    match = re.search(r"test_acc([\d.]+)", output)
    if match:
        return float(match.group(1)) * 100  # 转换为百分比形式
    else:
        print("未找到准确度输出！")
        return None


def main():
    num_runs = 5
    accuracies = []

    for i in range(num_runs):
        print(f"第 {i + 1} 次运行中...")
        acc = run_main_and_capture_accuracy()
        if acc is not None:
            accuracies.append(acc)
            print(f"本次准确度: {acc:.2f}%")
        else:
            print("运行失败，跳过本次结果")

    # 计算统计结果
    if len(accuracies) == 0:
        print("无有效结果")
        return

    mean_acc = statistics.mean(accuracies)
    std_dev = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0

    # 格式化输出
    acc_str = "、".join([f"{acc:.2f}%" for acc in accuracies])
    print(f"\n三次运行结果分别为：{acc_str}")
    print(f"平均结果为：{mean_acc:.2f}% ± {std_dev:.2f}%")


if __name__ == "__main__":
    main()