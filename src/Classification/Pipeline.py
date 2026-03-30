import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
from capymoa.evaluation import ClassificationEvaluator

class ClassificationExperimentRunner:
    def __init__(self, target_names=None):
        self.target_names = target_names if target_names is not None else ['Normal', 'Ataque']

    def _get_metric_class(self, metrics_dict, metric_name, target_class=1):
        norm_metrics = {str(k).lower(): v for k, v in metrics_dict.items()}
        metric_name = str(metric_name).lower()
        
        if target_class is None:
            val_0 = norm_metrics.get(f'{metric_name}_0', 0.0)
            val_1 = norm_metrics.get(f'{metric_name}_1', 0.0)
            
            val_0 = 0.0 if val_0 is None or np.isnan(val_0) else float(val_0)
            val_1 = 0.0 if val_1 is None or np.isnan(val_1) else float(val_1)
            
            return (val_0 + val_1) / 2.0
            
        else:
            val = norm_metrics.get(f'{metric_name}_{target_class}')
            return float(val) if val is not None and not np.isnan(val) else 0.0

    def _get_metric_classifier(self, y_true, y_pred, target_class):
        if len(y_true) == 0:
            return 0.0, 0.0, 0.0

        if target_class is None:
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        else:
            f1 = f1_score(y_true, y_pred, pos_label=target_class, average='binary', zero_division=0)
            prec = precision_score(y_true, y_pred, pos_label=target_class, average='binary', zero_division=0)
            rec = recall_score(y_true, y_pred, pos_label=target_class, average='binary', zero_division=0)
            
        return f1 * 100.0, prec * 100.0, rec * 100.0

    def print_metrics(self, results, target_class, warmup_instances=0):
        print("\n" + "="*80)
        print(f"{'RELATÓRIO ACUMULATIVO ':^80}")
        print("="*80)
        print(f"{'Modelo':<25} | {'Prec (%)':<10} | {'Rec (%)':<10} | {'F1 (%)':<10}")
        print("-" * 80)
        
        for name, data in results.items():
            # Aplica o slice para ignorar o período de warm-up nas métricas acumulativas
            y_true_list = data['y_true'][warmup_instances:] if len(data['y_true']) > warmup_instances else data['y_true']
            y_pred_list = data['y_pred'][warmup_instances:] if len(data['y_pred']) > warmup_instances else data['y_pred']

            f1, prec, rec = self._get_metric_classifier(y_true_list, y_pred_list, target_class)
            print(f"{name:<25} | {prec:>8.2f}   | {rec:>8.2f}   | {f1:>8.2f}")
        print("="*80 + "\n")

    def plot(self, results, attack_regions=None, title="Métricas", window_size=1000):
        fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
        (ax1, ax2, ax3) = axes
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        bg_colors = ['#F7C5CD', '#C5D9F7', '#C5F7C5', '#F7E6C5', '#E3C5F7', '#F7D9C5', '#C5F7E6']

        for i, (name, data) in enumerate(results.items()):
            color = colors[i % len(colors)]
            x_axis = data['instances']
            
            def clean(d_list): return [0.0 if (v is None or np.isnan(v)) else v for v in d_list]

            ax1.plot(x_axis, clean(data['precision']), label=name, color=color, alpha=0.85, linewidth=1.5, marker='o', zorder=3)
            ax2.plot(x_axis, clean(data['recall']), label=name, color=color, alpha=0.85, linewidth=1.5, marker='o', zorder=3)
            ax3.plot(x_axis, clean(data['f1']), label=name, color=color, alpha=0.85, linewidth=1.5, marker='o', zorder=3)

        added_attack_labels = set()
        for ax in axes:
            if attack_regions:
                for start, end, attack_idx in attack_regions:
                    attack_name = self.target_names[attack_idx] if attack_idx < len(self.target_names) else f'Ataque {attack_idx}'
                    bg_color = bg_colors[attack_idx % len(bg_colors)]
                    
                    label_to_show = f'{attack_name}' if attack_name not in added_attack_labels and ax == ax1 else ""
                    ax.axvspan(start, end, facecolor=bg_color, alpha=0.3, zorder=1, label=label_to_show)
                    
                    if label_to_show:
                        added_attack_labels.add(attack_name)
            
            ax.grid(True, alpha=0.3, linestyle=':', zorder=0)

        ax1.set_title(f"{title} (Resolução de {window_size} instâncias)", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Precision (%)", fontsize=12)
        ax2.set_ylabel("Recall (%)", fontsize=12)
        ax3.set_ylabel("F1-Score (%)", fontsize=12)
        ax3.set_xlabel("Instâncias", fontsize=14)

        handles, labels = ax1.get_legend_handles_labels()
        leg = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=len(results) + len(added_attack_labels), 
                         fontsize=12, frameon=False)
        for patch in leg.get_patches():
            patch.set_edgecolor('gray')
            patch.set_linewidth(1.0)
            patch.set_alpha(0.8)

        fig.subplots_adjust(bottom=0.15)
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        plt.show()

    def run_classification_evaluation(self, stream, algorithms, window_size=1000, title="Avaliação Prequencial", warmup_instances=0, target_class=1):
        results = {}
        attack_regions = []
        
        in_attack = False
        start_idx = 0
        current_attack_idx = None
        instance_idx = 0 
        
        for name in algorithms:
            results[name] = {
                'instances': [],
                'f1': [], 'precision': [], 'recall': [], 
                'y_true': [], 'y_pred': [],
                'evaluator': ClassificationEvaluator(schema=stream.get_schema())
            }

        stream.restart()

        while stream.has_more_instances():
            instance = stream.next_instance()
            true_label_multiclass = instance.y_index 
            
            if true_label_multiclass != 0: 
                if not in_attack:
                    in_attack = True
                    start_idx = instance_idx
                    current_attack_idx = true_label_multiclass
                elif true_label_multiclass != current_attack_idx:
                    attack_regions.append((start_idx, instance_idx - 1, current_attack_idx))
                    start_idx = instance_idx
                    current_attack_idx = true_label_multiclass
            else:
                if in_attack:
                    attack_regions.append((start_idx, instance_idx - 1, current_attack_idx))
                    in_attack = False
                    current_attack_idx = None
            
            binary_true_label = 1 if true_label_multiclass > 0 else 0

            for name, model in algorithms.items():
                res = results[name]
                evaluator = res['evaluator']

                prediction = model.predict(instance)
                if prediction is None:
                    prediction = 0
                
                # Só computa métricas no CapyMOA se o warmup tiver passado
                if instance_idx >= warmup_instances:
                    evaluator.update(true_label_multiclass, prediction)
                
                binary_prediction = 1 if prediction > 0 else 0
                res['y_true'].append(binary_true_label)
                res['y_pred'].append(binary_prediction)

                model.train(instance)

                # Coleta as métricas do gráfico ignorando o warmup
                if instance_idx >= warmup_instances and instance_idx > 0 and instance_idx % window_size == 0:
                    res['instances'].append(instance_idx)
                    m_dict = evaluator.metrics_dict()
                    
                    res['precision'].append(self._get_metric_class(m_dict, 'precision', target_class))
                    res['recall'].append(self._get_metric_class(m_dict, 'recall', target_class))
                    res['f1'].append(self._get_metric_class(m_dict, 'f1_score', target_class))

            instance_idx += 1
        
        if in_attack:
            attack_regions.append((start_idx, instance_idx - 1, current_attack_idx))
            
        self.print_metrics(results, target_class, warmup_instances)
        self.plot(results, attack_regions=attack_regions, title=title, window_size=window_size)