import pandas as pd
import numpy as np
import os
import glob
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

class ScenarioGenerator:
    """
    classe responsável por orquestrar a geração de cenários de rede,
    mesclando um baseline benigno com fatias de arquivos de ataque,
    e fornecendo ferramentas de visualização para fluxos de dados.
    """
    
    def __init__(self, input_folder, output_path, baseline_file, target_files=None, n_benign_samples=None, logging=True, sort_by_timestamp=True, remove_duplicates=True):
        self.input_folder = input_folder
        self.output_path = output_path
        self.baseline_file = baseline_file
        self.target_files = target_files
        self.n_benign_samples = n_benign_samples
        self.logging = logging
        self.sort_by_timestamp = sort_by_timestamp
        self.remove_duplicates = remove_duplicates
        
        self.benign_df = pd.DataFrame()
        self.total_benign = 0
        self.final_df = pd.DataFrame()
        
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.baseline_file), exist_ok=True)

    def _log(self, message):
        if self.logging:
            print(message)

    def _prepare_baseline(self):
        if os.path.exists(self.baseline_file):
            self._log("\n[*] Baseline BENIGN encontrado! Carregando de forma rápida...")
            self.benign_df = pd.read_csv(self.baseline_file, low_memory=False)
            self.total_benign = len(self.benign_df)
            self._log(f"    -> {self.total_benign} amostras BENIGN carregadas da memória.")
        else:
            self._log(f"\n[*] Baseline não encontrado. Processando arquivos da pasta {self.input_folder}...")
            
            if self.target_files:
                self._log(f"    -> Filtro ativo: Lendo apenas os {len(self.target_files)} arquivos especificados.")
                all_files = [os.path.join(self.input_folder, f) for f in self.target_files]
                all_files = [f for f in all_files if os.path.exists(f)] 
            else:
                self._log("    -> Lendo todos os arquivos CSV da pasta.")
                all_files = glob.glob(os.path.join(self.input_folder, "*.csv"))
                
            benign_data = []
            
            for filename in all_files:
                self._log(f"    Lendo: {os.path.basename(filename)}...")
                try:
                    for chunk in pd.read_csv(filename, chunksize=100000, low_memory=False, encoding_errors='ignore', on_bad_lines='skip'):
                        chunk.columns = chunk.columns.str.strip()
                        if 'Label' not in chunk.columns: continue
                        
                        # Adicionado .str.strip() para limpar espaços em branco invisíveis do CIC-IDS-2017
                        benign_chunk = chunk[chunk['Label'].astype(str).str.strip().str.upper() == 'BENIGN']
                        
                        if not benign_chunk.empty:
                            benign_data.append(benign_chunk)
                except Exception as e:
                    self._log(f"    [ERRO] Falha ao ler {os.path.basename(filename)}: {e}")

            if not benign_data:
                self._log("    [!] Nenhuma amostra benigna foi encontrada. Encerrando.")
                return

            self._log("    Concatenando DataFrame e ordenando por Timestamp...")
            self.benign_df = pd.concat(benign_data, ignore_index=True)
            
            if 'Timestamp' in self.benign_df.columns:
                try:
                    self.benign_df['Timestamp'] = pd.to_datetime(self.benign_df['Timestamp'], errors='coerce')
                    self.benign_df = self.benign_df.sort_values(by='Timestamp').reset_index(drop=True)
                except:
                    pass

            self.total_benign = len(self.benign_df)
            self._log(f"    Salvando {self.total_benign} amostras no arquivo base...")
            self.benign_df.to_csv(self.baseline_file, index=False)
    def _reduce_benign_stratified(self):
        # se não passou parâmetro ou pediu mais do que tem, mantém o dataframe original
        if self.n_benign_samples is None or self.n_benign_samples >= len(self.benign_df):
            self._log(f"\n[*] Redução ignorada. Mantendo todas as {len(self.benign_df)} amostras benignas originais.")
            return
        
        self._log(f"\n[*] Reduzindo amostras benignas para {self.n_benign_samples} de forma estratificada...")
        df = self.benign_df
        n_samples = self.n_benign_samples
        
        # procura a coluna de porta de destino para basear a estratificação
        stratify_col = None
        for col in df.columns:
            if 'destination port' in col.lower():
                stratify_col = col
                break
                
        # fallback: tenta usar o protocolo se não achar porta
        if not stratify_col:
            for col in df.columns:
                if 'protocol' in col.lower():
                    stratify_col = col
                    break
                    
        if not stratify_col:
            self._log("    [AVISO] Coluna de porta ou protocolo não encontrada. Reduzindo de forma aleatória simples.")
            reduced_df = df.sample(n=n_samples, random_state=42)
        else:
            self._log(f"    -> Estratificando com base na coluna: '{stratify_col}'")
            
            # agrupa pelas portas e extrai amostras mantendo a proporção
            reduced_df = df.groupby(stratify_col, group_keys=False).apply(
                lambda x: x.sample(
                    n=int(np.round(len(x) / len(df) * n_samples)), 
                    random_state=42, 
                    replace=True if len(x) < int(np.round(len(x) / len(df) * n_samples)) else False
                )
            )
            
            # ajusta sobras ou faltas matemáticas
            current_len = len(reduced_df)
            if current_len < n_samples:
                diff = n_samples - current_len
                extra_samples = df.drop(reduced_df.index).sample(n=diff, random_state=42)
                reduced_df = pd.concat([reduced_df, extra_samples])
            elif current_len > n_samples:
                reduced_df = reduced_df.sample(n=n_samples, random_state=42)
                
        # reordena a linha do tempo
        if 'Timestamp' in reduced_df.columns:
            try:
                reduced_df['Timestamp'] = pd.to_datetime(reduced_df['Timestamp'], errors='coerce')
                reduced_df = reduced_df.sort_values(by='Timestamp').reset_index(drop=True)
            except:
                pass
                
        self.benign_df = reduced_df
        self.total_benign = len(self.benign_df) # atualiza a contagem oficial da classe
        self._log(f"    -> Redução concluída com sucesso! Total final: {self.total_benign} amostras benignas.")

    def _collect_attacks(self, attack_config):
        attack_blocks = []
        iterator_cache = {} 
        
        self._log("\n[*] Coletando fatias de ataque (Linha a Linha)...")
        for config_item in attack_config:
            
            # desempacotamento flexível para aceitar o filtro de labels
            if len(config_item) == 3:
                target_file, n_samples, target_labels = config_item
                if isinstance(target_labels, str):
                    target_labels = [target_labels]
            else:
                target_file, n_samples = config_item
                target_labels = None

            attack_path = os.path.join(self.input_folder, target_file)
            
            if not os.path.exists(attack_path):
                self._log(f"    [ERRO] Arquivo não encontrado: {target_file}")
                continue

            if attack_path not in iterator_cache:
                iterator = pd.read_csv(attack_path, chunksize=100000, low_memory=False, encoding_errors='ignore', on_bad_lines='skip')
                iterator_cache[attack_path] = {'iterator': iterator, 'buffer': pd.DataFrame()}
                
            state = iterator_cache[attack_path]
            collected_samples = []
            total_collected = 0
            
            while total_collected < n_samples:
                # tenta extrair amostras do buffer
                if not state['buffer'].empty:
                    if target_labels is not None:
                        mask = state['buffer']['Label'].isin(target_labels)
                        valid_buffer = state['buffer'][mask]
                    else:
                        valid_buffer = state['buffer']

                    qty_to_fetch = min(n_samples - total_collected, len(valid_buffer))
                    
                    if qty_to_fetch > 0:
                        # extrai exatamente os índices válidos
                        indices_to_take = valid_buffer.index[:qty_to_fetch]
                        collected_samples.append(state['buffer'].loc[indices_to_take])
                        total_collected += qty_to_fetch
                        
                        # remove do buffer APENAS as linhas que já coletamos
                        state['buffer'] = state['buffer'].drop(indices_to_take).reset_index(drop=True)
                        
                # se ainda precisa de amostras, avança a leitura no disco
                if total_collected < n_samples:
                    try:
                        chunk = next(state['iterator'])
                        chunk.columns = chunk.columns.str.strip()
                        
                        if 'Label' not in chunk.columns: continue
                        
                        # adiciona o novo chunk ao buffer e o loop repete a extração
                        state['buffer'] = pd.concat([state['buffer'], chunk], ignore_index=True)
                        
                    except StopIteration:
                        self._log(f"\n    [AVISO] Limite do arquivo '{target_file}' atingido!")
                        self._log(f"            -> Solicitado: {n_samples} | Encontrado: {total_collected}")
                        self._log(f"            -> O cenário continuará utilizando as {total_collected} amostras obtidas.\n")
                        break 
                    
            # finaliza o bloco e salva o que conseguiu coletar
            if collected_samples:
                final_attack_block = pd.concat(collected_samples, ignore_index=True)
                attack_blocks.append(final_attack_block)
                
                # log detalhado informando o filtro
                if target_labels:
                    labels_str = ", ".join(target_labels)
                    self._log(f"    -> Bloco montado: {len(final_attack_block)} amostras de {target_file} (Filtro: {labels_str}).")
                else:
                    self._log(f"    -> Bloco montado: {len(final_attack_block)} amostras de {target_file} (Sem filtro).")
            else:
                self._log(f"    [AVISO] Nenhuma amostra da classe desejada foi encontrada em {target_file}.")
                
        return attack_blocks

    def _assemble_and_save(self, attack_blocks):
        n_injections = len(attack_blocks)
        
        if n_injections == 0:
            self._log("\n[!] Nenhum ataque válido carregado.")
            return

        benign_slice_size = self.total_benign // (n_injections + 1)
        
        self._log("\n[*] Montando fluxo intercalado...")
        self._log(f"    Dividindo {self.total_benign} BENIGN em {n_injections + 1} fatias de aprox. {benign_slice_size} linhas.")
        
        final_stream = []
        current_idx = 0
        
        for i, atk_block in enumerate(attack_blocks):
            next_idx = current_idx + benign_slice_size
            final_stream.append(self.benign_df.iloc[current_idx:next_idx])
            current_idx = next_idx
            final_stream.append(atk_block)
            
        final_stream.append(self.benign_df.iloc[current_idx:])
        final_df = pd.concat(final_stream, ignore_index=True)
        
        # --- remoção de duplicatas ---
        if self.remove_duplicates:
            size_before = len(final_df)
            final_df = final_df.drop_duplicates(keep='first').reset_index(drop=True)
            removed_count = size_before - len(final_df)
            if removed_count > 0:
                self._log(f"    -> Removidas {removed_count} linhas duplicadas.")
                
        # --- ordenação por timestamp ---
        if self.sort_by_timestamp and 'Timestamp' in final_df.columns:
            self._log("    -> Ordenando cenário por Timestamp...")
            try:
                final_df['Timestamp'] = pd.to_datetime(final_df['Timestamp'], errors='coerce')
                final_df = final_df.sort_values(by='Timestamp').reset_index(drop=True)
            except Exception as e:
                self._log(f"    [AVISO] Falha ao ordenar por Timestamp: {e}")

        self.final_df = final_df
        self._log("    Salvando no disco...")
        final_df.to_csv(self.output_path, index=False)
        
        self._log("\n" + "="*60)
        self._log(" SUCESSO! Cenário final salvo e pronto para Análise.")
        self._log("-" * 60)
        self._log(final_df['Label'].value_counts().to_string())
        self._log("="*60)

    def generate(self, attack_config):
        self._prepare_baseline()
        self._reduce_benign_stratified() 
        attack_blocks = self._collect_attacks(attack_config)
        self._assemble_and_save(attack_blocks)

    def plot_scenario(self, features_plot=['Total Fwd Packets', 'Total Backward Packets'], window_size=50):
        if self.final_df.empty:
            print("[!] Gere o cenário primeiro chamando a função gerar().")
            return

        df_plot = self.final_df.reset_index(drop=True)
        valid_columns = [col for col in features_plot if col in df_plot.columns]
        
        if not valid_columns:
            print(f"[!] Nenhuma das features fornecidas foi encontrada no dataset. Features disponíveis: {list(df_plot.columns[:5])}...")
            return

        plt.figure(figsize=(16, 7))
        
        # identifica blocos de ataque para o fundo colorido
        df_plot['id_block'] = df_plot['Label'].ne(df_plot['Label'].shift()).cumsum()
        unique_attacks = [lbl for lbl in df_plot['Label'].unique() if lbl != 'BENIGN']
        highlight_colors = ["#FFB4CA", "#54E5F5", "#F8DE4C", "#B0FFB0", "#DA9EFF", "#FFCDAB"]
        color_map = {attack: highlight_colors[i % len(highlight_colors)] for i, attack in enumerate(unique_attacks)}
        
        for _, block in df_plot.groupby('id_block'):
            current_label = block['Label'].iloc[0]
            if current_label != 'BENIGN':
                start_idx = block.index[0]
                end_idx = block.index[-1]
                # pinta o fundo com alpha para transparência e zorder=1 para ficar no fundo
                plt.axvspan(start_idx, end_idx, color=color_map[current_label], alpha=0.2, zorder=1)

        # plota as features com média móvel (zorder=3 -> na frente do fundo)
        line_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#8c564b'] 
        
        for i, col in enumerate(valid_columns):
            # transforma a coluna para numérico (caso haja algum resquício de string)
            df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')
            
            # aplica a média móvel. min_periods=1 evita que os primeiros 49 pacotes fiquem vazios (nan)
            moving_average = df_plot[col].rolling(window=window_size, min_periods=1).mean()
            
            plt.plot(df_plot.index, moving_average + 1, label=f'{col} (MA {window_size})', 
                     linewidth=1.0, color=line_colors[i % len(line_colors)], zorder=3)
                
        # cria uma legenda organizada
        handles, labels = plt.gca().get_legend_handles_labels()
        
        # adiciona os patches de cores de fundo na legenda
        for attack, color in color_map.items():
            patch = mpatches.Patch(color=color, alpha=0.5, label=f'{attack}')
            handles.append(patch)
            
        plt.yscale('log')
        plt.title(f'Fluxo Temporal - Média Móvel {window_size} janelas', fontsize=15, fontweight='bold')
        plt.xlabel('Índice da Amostra', fontsize=12)
        plt.ylabel('Valor da Feature', fontsize=12)
        plt.grid(True, which="major", ls="--", alpha=0.4, zorder=2)
        
        # posiciona a legenda fora do gráfico para não cobrir as linhas
        plt.legend(handles=handles)
        plt.tight_layout()
        plt.show()