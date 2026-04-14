import pandas as pd
import os
import glob
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list

warnings.filterwarnings('ignore')

class DatasetHandler:
    def __init__(self, logging=True):
        self.logging = logging

    def _log(self, message):
        if self.logging:
            print(message)

    def create_balanced_dataset(
        self, 
        src_dir, 
        dest_dir, 
        output_filename, 
        n_samples_per_class, 
        chunk_size=100000, 
        target_files=None,
        ignored_classes=None, 
        allow_insufficient=False
    ):
        
        # inicializa lista de classes ignoradas
        if ignored_classes is None: ignored_classes = []
        if not os.path.exists(dest_dir): os.makedirs(dest_dir)
        
        # define quais arquivos processar
        if target_files:
            csv_files = [os.path.join(src_dir, f) for f in target_files]
            csv_files = [f for f in csv_files if os.path.exists(f)]
        else:
            csv_files = glob.glob(os.path.join(src_dir, "*.csv"))
        
        if not csv_files:
            self._log(f" [ERRO] Nenhum arquivo CSV válido encontrado em {src_dir}")
            return

        full_output_path = os.path.join(dest_dir, output_filename)
        
        if os.path.exists(full_output_path):
            os.remove(full_output_path)

        self._log("="*80)
        self._log(f"PROCESSAMENTO OTIMIZADO (CHUNKS): {output_filename}")
        self._log(f"Arquivos Selecionados: {len(csv_files)}")
        self._log(f"Tamanho do Lote (Chunksize): {chunk_size}")
        if ignored_classes:
            self._log(f"Classes Ignoradas: {ignored_classes}")
        self._log("="*80)

        # varredura global lendo em lotes
        self._log("\n[*] Varredura global (Lendo em lotes)...")
        
        global_counts = {}
        
        for file in csv_files:
            try:
                # ignorando erros de encoding e bad lines
                header_df = pd.read_csv(file, nrows=0, encoding_errors='ignore', on_bad_lines='skip')
                col_label = next((col for col in header_df.columns if 'label' in col.lower()), None)
                
                if not col_label: continue

                # ignorando erros de encoding e bad lines nos chunks
                for chunk in pd.read_csv(file, usecols=[col_label], chunksize=chunk_size, encoding_errors='ignore', on_bad_lines='skip'):
                    chunk.columns = ['Label']
                    counts = chunk['Label'].value_counts().to_dict()
                    
                    for label_class, qty in counts.items():
                        if label_class not in ignored_classes:
                            global_counts[label_class] = global_counts.get(label_class, 0) + qty
                            
            except Exception as e:
                self._log(f" [!] Erro ao ler {os.path.basename(file)}: {e}")

        # validação da quantidade de amostras solicitadas
        self._log("[*] Validando quantidades disponíveis...")

        insufficient_classes = {}
        classes_to_collect = []

        for label_class, total in global_counts.items():
            if total < n_samples_per_class:
                insufficient_classes[label_class] = total
            else:
                classes_to_collect.append(label_class)

        if insufficient_classes:
            if not allow_insufficient:
                self._log("\n" + "!"*80)
                self._log(" [ERRO CRÍTICO] Classes insuficientes detectadas (Processo Interrompido):")
                for cls, total in insufficient_classes.items():
                    self._log(f"   -> {cls}: {total} (Meta: {n_samples_per_class})")
                self._log("!"*80)
                return
            else:
                # mostrando quais são as classes insuficientes permitidas
                insufficient_names = list(insufficient_classes.keys())
                self._log(f" [AVISO] Permitindo {len(insufficient_classes)} classes insuficientes: {insufficient_names}")
                classes_to_collect.extend(insufficient_classes.keys())
        
        if not classes_to_collect:
            self._log(" [ERRO] Nenhuma classe para coletar.")
            return

        # coleta e escrita com buffer
        self._log(f"\n[*] Coletando e Salvando em disco (Lotes de {chunk_size})...")
        
        collected_samples_count = {cls: 0 for cls in classes_to_collect}
        
        buffer_list = []
        buffer_row_count = 0
        buffer_limit = chunk_size 
        first_write = True 

        for file in csv_files:
            all_done = True
            for cls in classes_to_collect:
                target = min(n_samples_per_class, global_counts[cls])
                if collected_samples_count[cls] < target:
                    all_done = False
                    break
            if all_done: break

            self._log(f"   -> Processando: {os.path.basename(file)}")

            try:
                # ignorando erros de encoding e bad lines nos chunks reais
                for chunk in pd.read_csv(file, chunksize=chunk_size, encoding_errors='ignore', on_bad_lines='skip'):
                    chunk.columns = chunk.columns.str.strip()
                    
                    if 'Label' not in chunk.columns: continue

                    chunk_selection = [] 

                    for label_class in classes_to_collect:
                        target = min(n_samples_per_class, global_counts[label_class])
                        current = collected_samples_count[label_class]
                        
                        if current >= target:
                            continue
                        
                        class_chunk_df = chunk[chunk['Label'] == label_class]
                        
                        if class_chunk_df.empty:
                            continue

                        remaining = target - current
                        n_to_select = min(remaining, len(class_chunk_df))
                        
                        samples = class_chunk_df.sample(n=n_to_select, random_state=42)
                        chunk_selection.append(samples)
                        collected_samples_count[label_class] += n_to_select
                    
                    if chunk_selection:
                        chunk_merged = pd.concat(chunk_selection)
                        buffer_list.append(chunk_merged)
                        buffer_row_count += len(chunk_merged)

                    if buffer_row_count >= buffer_limit:
                        df_buffer = pd.concat(buffer_list)
                        
                        mode = 'w' if first_write else 'a'
                        header = first_write 
                        
                        df_buffer.to_csv(full_output_path, mode=mode, header=header, index=False)
                        
                        self._log(f"      [IO] Buffer cheio ({buffer_row_count} linhas). Salvando lote no disco...")
                        
                        buffer_list = []
                        buffer_row_count = 0
                        first_write = False

            except Exception as e:
                self._log(f" [!] Erro ao processar chunk em {file}: {e}")

        # escrita com o que ficou no buffer
        if buffer_list:
            df_buffer = pd.concat(buffer_list)
            mode = 'w' if first_write else 'a'
            header = first_write
            df_buffer.to_csv(full_output_path, mode=mode, header=header, index=False)
            self._log(f"      [IO] Salvando lote final ({len(df_buffer)} linhas)...")

        self._log("\n" + "="*80)
        self._log("CONCLUÍDO COM SUCESSO")
        self._log(f"Arquivo gerado: {full_output_path}")
        self._log("="*80)

    def sort_dataset_by_timestamp(self, file_path):
        if not os.path.exists(file_path):
            self._log(f" [ERRO] Arquivo não encontrado: {file_path}")
            return
            
        self._log("="*80)
        self._log(f"INICIANDO ORDENAÇÃO CRONOLÓGICA: {os.path.basename(file_path)}")
        self._log("="*80)
            
        try:
            # lê o arquivo 
            df = pd.read_csv(file_path, encoding_errors='ignore', on_bad_lines='skip')
            
            # limpa os nomes das colunas
            df.columns = df.columns.str.strip()
            
            # identifica dinamicamente a coluna de tempo
            timestamp_col = None
            for col in df.columns:
                if 'timestamp' in col.lower() or 'time' in col.lower() or 'date' in col.lower():
                    timestamp_col = col
                    break
                    
            if not timestamp_col:
                self._log(" [ERRO] Nenhuma coluna de data/hora (Timestamp) foi encontrada no dataset.")
                return
                
            self._log(f" [INFO] Coluna de tempo identificada: '{timestamp_col}'. Convertendo dados...")
            
            # converte a coluna para o tipo datetime do pandas
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
            
            self._log(" [INFO] Ordenando as amostras...")
            df = df.sort_values(by=timestamp_col)
            
            # sobrescreve o mesmo arquivo
            self._log(f" [INFO] Salvando atualizações no arquivo: {file_path}...")
            df.to_csv(file_path, index=False)
            
            # exibe a análise final
            self._log("\n" + "="*50)
            self._log(" ORDENAÇÃO CONCLUÍDA COM SUCESSO")
            self._log("="*50)
            if 'Label' in df.columns:
                self._log(f"Total de Amostras no Arquivo: {len(df)}")
                self._log("-" * 30)
                self._log("Contagem por Rótulo (Label):")
                self._log(df['Label'].value_counts())
            else:
                self._log(" [AVISO] Coluna 'Label' não encontrada para realizar a contagem.")
            self._log("="*50)
                
        except Exception as e:
            self._log(f" [!] Ocorreu um erro durante a ordenação: {e}")

    def extract_ovr_feature_importance(self, X, y, target_names, top_per_class=10):
        self._log("\n" + "="*80)
        self._log(" INICIANDO EXTRAÇÃO DE FEATURES OVR (ONE-VS-All)")
        self._log("="*80)

        features = X.columns.tolist()
        class_top_features = {}
        all_important_features = set()
        
        # Identifica as classes únicas presentes no y
        unique_classes = np.unique(y)

        for cls_idx in unique_classes:
            cls_name = target_names[cls_idx]
            self._log(f" [*] Analisando Assinatura Local: {cls_name} vs Resto do Tráfego...")
            
            # Cria alvo binário (1 para a classe atual, 0 para o resto)
            y_binary = (y == cls_idx).astype(int)

            # Random Forest focado apenas nesta separação
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y_binary)

            # Extração e ordenação das importâncias
            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1]

            # Seleciona as N melhores features para ESTE ataque
            top_n_features = [features[i] for i in indices[:top_per_class]]
            class_top_features[cls_name] = top_n_features
            
            # Adiciona ao conjunto global
            all_important_features.update(top_n_features)

        consolidated_features = list(all_important_features)
        
        self._log("\n" + "-"*50)
        self._log(f" [RESUMO] Total de classes analisadas: {len(unique_classes)}")
        self._log(f" [RESUMO] Features únicas consolidadas: {len(consolidated_features)} (de {len(features)} originais)")
        self._log("-"*50)

        return class_top_features, consolidated_features

    def plot_similarity_and_feature_groups(self, X, y, target_names, selected_features, metric='euclidean', linkage_method='ward'):
        # Filtra as features e calcula assinaturas
        X_filtered = X[selected_features].copy()
        X_filtered['Label_Idx'] = y
        
        df_signature = X_filtered.groupby('Label_Idx').mean()
        df_signature.index = [target_names[idx] for idx in df_signature.index]

        # Normalização
        scaler = MinMaxScaler()
        df_sig_norm = pd.DataFrame(
            scaler.fit_transform(df_signature), 
            index=df_signature.index, 
            columns=df_signature.columns
        )

        if linkage_method == 'ward' and metric != 'euclidean':
            self._log(f" [AVISO] O método 'ward' exige distância 'euclidean'. Mudando automaticamente.")
            metric = 'euclidean'

        # Clusterização
        # Ataques (Linhas)
        dist_matrix_rows = pdist(df_sig_norm, metric=metric)
        Z_rows = linkage(dist_matrix_rows, method=linkage_method)
        ordem_ataques = leaves_list(Z_rows) 

        # Features (Colunas)
        dist_matrix_cols = pdist(df_sig_norm.T, metric=metric)
        Z_cols = linkage(dist_matrix_cols, method=linkage_method)
        ordem_features = leaves_list(Z_cols)

        # Reordenando os dados para que os parecidos fiquem juntos
        df_sorted = df_sig_norm.iloc[ordem_ataques, ordem_features]

        # DENDROGRAMA PRINCIPAL
        plt.figure(figsize=(14, 6))
        dendrogram(Z_rows, labels=df_sig_norm.index.tolist(), leaf_rotation=45, leaf_font_size=14)
        plt.ylabel(f"Distance {metric}", fontsize=18)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

        # MAPA DE CALOR 
        plt.figure(figsize=(16, 8))
        
        ax = sns.heatmap(
            df_sorted, 
            cmap='viridis', 
            annot=False, 
            linewidths=0.5,
            cbar_kws={"shrink": 0.5},
        )

        # Personalização da barra de cores
        cbar = ax.collections[0].colorbar
        cbar.set_label("Normalized Intensity", fontsize=16, labelpad=15)
        
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(rotation=0, fontsize=14)
        plt.xlabel("Selected Features", fontsize=16, labelpad=10)
        plt.ylabel("Attacks", fontsize=16, labelpad=10)
        
        plt.tight_layout()
        plt.show()

        print("features = [")
        for col in df_sorted.columns:
            print(f"    '{col}',")
        print("]")