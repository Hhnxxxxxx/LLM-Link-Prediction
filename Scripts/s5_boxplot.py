from statistics import mean, stdev
import matplotlib.pyplot as plt
from s3_build_graph import build_temporal_graph
from s4_gcn_training import eval_auc_ap, train, device


def run_multiple_trials(device, n_runs=10):
    results = {"BoW+DP": {"auc": [], "ap": []},
               "SciBERT+MLP": {"auc": [], "ap": []}}

    for i in range(n_runs):
        print(f"BoW+DP Run {i + 1}/{n_runs}")
        data = build_temporal_graph("openalex_9000.json", "bow_embeddings_9000.npy", cutoff=2018)
        model, decoder, test_data = train("BoW", device=device, decoder_type="dp", epochs=200, data=data)
        auc, ap = eval_auc_ap(model, decoder, test_data)
        results["BoW+DP"]["auc"].append(auc)
        results["BoW+DP"]["ap"].append(ap)

    for i in range(n_runs):
        print(f"SciBERT+MLP Run {i + 1}/{n_runs}")
        data = build_temporal_graph("openalex_9000.json", "sci_bert_embeddings_9000.npy", cutoff=2018)
        model, decoder, test_data = train("SciBERT", device=device, decoder_type="mlp", epochs=200, data=data)
        auc, ap = eval_auc_ap(model, decoder, test_data)
        results["SciBERT+MLP"]["auc"].append(auc)
        results["SciBERT+MLP"]["ap"].append(ap)

    # Print means and stds
    for key in results:
        aucs = results[key]["auc"]
        aps = results[key]["ap"]
        print(f"\n{key}:")
        print(f"AUC  mean = {mean(aucs):.4f}, std = {stdev(aucs):.4f}")
        print(f"AP   mean = {mean(aps):.4f}, std = {stdev(aps):.4f}")

    # Boxplot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.boxplot([results["BoW+DP"]["auc"], results["SciBERT+MLP"]["auc"]],
                labels=["BoW+DP", "SciBERT+MLP"], patch_artist=True, showmeans=True)
    plt.title("AUC Boxplot")
    plt.ylabel("AUC")

    plt.subplot(1, 2, 2)
    plt.boxplot([results["BoW+DP"]["ap"], results["SciBERT+MLP"]["ap"]],
                labels=["BoW+DP", "SciBERT+MLP"], patch_artist=True, showmeans=True)
    plt.title("AP Boxplot")
    plt.ylabel("Average Precision")

    plt.tight_layout()
    plt.show()

    return results


run_multiple_trials(device, n_runs=50)
