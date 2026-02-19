from tbparse import SummaryReader
import matplotlib.pyplot as plt
import os

# log_dir = "../logs_mrd/baseline_google-embeddinggemma-300m_poolmean/version_6"
# log_dir = "../logs_mrd/baseline_google-embeddinggemma-300m_poolcls/version_0"
# log_dir = "../logs_mrd/baseline_google-embeddinggemma-300m_poolmean/version_4"
# log_dir = "../logs_mrd/cDGM_google-embeddinggemma-300m_k15_gcn_euclidean_poolmean/version_0"
# baseline_distilbert-base-uncased_poolmean
# cdgm_google-embeddinggemma-300m_k5_gat_euclidean_poolmean
# dDGM_distilbert-base-uncased_k5_gat_euclidean_poolmean
# dDGM_google-embeddinggemma-300m_k5_gat_euclidean_poolmean
# "../logs_20news/dDGM_distilbert-base-uncased_k10_gat_euclidean_poolmean/version_21"
# "../logs_20news/cDGM_distilbert-base-uncased_k15_gcn_euclidean_poolmean/version_1"
# "../logs_20news/dDGM_google-embeddinggemma-300m_k5_gat_euclidean_poolmean/version_4"
log_dir = "../logs_20news/dDGM_google-embeddinggemma-300m_k10_gat_euclidean_poolmean/version_2"
reader = SummaryReader(log_dir)
df = reader.scalars

output_dir = os.path.join(log_dir, 'log_figures')
os.makedirs(output_dir, exist_ok=True)

train_metrics = ['train_loss', 'train_acc', 'train_graph_loss']
val_metrics = ['val_loss', 'val_acc']


def plot_metric(metric_name, df, output_dir):
    metric_df = df[df['tag'] == metric_name].copy()

    if len(metric_df) == 0:
        print(f"No data for {metric_name}")
        return

    if 'train' in metric_name:
        # smoothing
        window = max(10, len(metric_df) // 50)
        metric_df['smoothed'] = metric_df['value'].rolling(
            window=window, center=True).mean()

        plt.figure(figsize=(10, 6))
        plt.plot(metric_df['step'], metric_df['value'],
                 alpha=0.3, label='raw', linewidth=0.5)
        plt.plot(metric_df['step'], metric_df['smoothed'],
                 label='smoothed', linewidth=2)
        plt.legend()
    else:
        plt.figure(figsize=(10, 6))
        plt.plot(metric_df['step'], metric_df['value'],
                 marker='o', linewidth=2)

    plt.title(f'{metric_name} over training', fontsize=14)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f'{metric_name}.png')
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"✓ Saved {metric_name}: {len(metric_df)} points")
    print(f"  Final value: {metric_df['value'].iloc[-1]:.4f}")


all_tags = df['tag'].unique()
print(f'\n--- Found {len(all_tags)} metrics: {list(all_tags)}\n')

for tag in all_tags:
    if tag != 'epoch' and tag != 'hp_metric':
        plot_metric(tag, df, output_dir)

print('\n--- Creating comparison plots ---')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

train_loss = df[df['tag'] == 'train_loss'].copy()
if len(train_loss) > 0:
    window = max(10, len(train_loss) // 50)
    train_loss['smoothed'] = train_loss['value'].rolling(
        window=window, center=True).mean()
    ax1.plot(train_loss['step'], train_loss['smoothed'],
             label='Train Loss', linewidth=2)

val_loss = df[df['tag'] == 'val_loss']
if len(val_loss) > 0:
    ax1.plot(val_loss['step'], val_loss['value'],
             label='Val Loss', marker='o', linewidth=2)

ax1.set_xlabel('Step')
ax1.set_ylabel('Loss')
ax1.set_title('Loss Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)

train_acc = df[df['tag'] == 'train_acc'].copy()
if len(train_acc) > 0:
    window = max(10, len(train_acc) // 50)
    train_acc['smoothed'] = train_acc['value'].rolling(
        window=window, center=True).mean()
    ax2.plot(train_acc['step'], train_acc['smoothed'],
             label='Train Acc', linewidth=2)

val_acc = df[df['tag'] == 'val_acc']
if len(val_acc) > 0:
    ax2.plot(val_acc['step'], val_acc['value'],
             label='Val Acc', marker='o', linewidth=2)

ax2.set_xlabel('Step')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Accuracy Comparison')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'comparison.png'), dpi=150)
plt.close()

for metric in ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc', 'test_edge_prob_mean', 'train_edge_prob_mean']:
    metric_df = df[df['tag'] == metric]
    if len(metric_df) > 0:
        print(f'\n{metric}:')
        print(f'  Start: {metric_df["value"].iloc[0]:.4f}')
        print(f'  End:   {metric_df["value"].iloc[-1]:.4f}')
        print(
            f'  Best:  {metric_df["value"].max():.4f}' if 'acc' in metric else f'  Best:  {metric_df["value"].min():.4f}')
