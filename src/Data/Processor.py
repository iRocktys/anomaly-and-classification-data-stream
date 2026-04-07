import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from capymoa.stream import NumpyStream

class DataStreamProcessor:
    def __init__(self, logging=True, selected_features=None):
        self.logging = logging
        self.selected_features = selected_features

    def _log(self, message):
        if self.logging:
            print(message)

    def _remove_features(self, X, y, threshold_var=None, threshold_corr=None, top_n_features=None):
        initial_count = X.shape[1]
        self._log(f"\n--- Iniciando Processo de Seleção de Features (Total: {initial_count}) ---")

        # remoção por variância 
        if threshold_var is not None:
            selector = VarianceThreshold(threshold=threshold_var)
            selector.fit(X)
            cols_var = X.columns[selector.get_support()]
            removed_count = initial_count - len(cols_var)
            X = X[cols_var]
            self._log(f"Variância: {removed_count} features removidas. Restantes: {X.shape[1]}")
        else:
            self._log("Remoção de Variância: Pular.")

        # remoção por correlação de pearson 
        if threshold_corr is not None:
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > threshold_corr)]
            X = X.drop(columns=to_drop)
            self._log(f"Correlação (>{threshold_corr}): {len(to_drop)} features redundantes removidas. Restantes: {X.shape[1]}")
        else:
            self._log("Remover Correlação: Pular.")

        # random forest importance
        if top_n_features is not None:
            if X.shape[1] > top_n_features:
                rf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
                rf.fit(X, y)
                importances = pd.Series(rf.feature_importances_, index=X.columns)
                selected_feats = importances.nlargest(top_n_features).index.tolist()
                X = X[selected_feats]
                self._log(f"Random Forest: Top {top_n_features} selecionadas.")
            else:
                self._log("Random Forest: Ignorado (Features atuais <= Top N).")
        else:
            self._log("Random Forest: Pular.")

        self._log(f"Features Finais ({X.shape[1]}) - {X.columns.tolist()}")
        self._log("--- Fim do Processo de Seleção de Features ---\n")

        return X

    def _normalize_data(self, X, method=None): 
        match method:
            case "MinMaxScaler":
                scaler = MinMaxScaler()
            case "StandardScaler":
                scaler = StandardScaler()
            case "RobustScaler":
                scaler = RobustScaler()
            case _:
                self._log("Normalização: Dados originais mantidos.")
                return X.values if hasattr(X, 'values') else X

        # aplica a transformação se encontrou um método válido
        scaled_x = scaler.fit_transform(X)
        self._log(f"Normalização: {method}")
        return scaled_x

    def _handle_missing_values(self, X, method='0'):
        match str(method).lower():
            case 'media':
                self._log("Tratamento de Nulos: Preenchendo com a MÉDIA das colunas...")
                return X.fillna(X.mean())
            case 'mediana':
                self._log("Tratamento de Nulos: Preenchendo com a MEDIANA das colunas...")
                return X.fillna(X.median())
            case 'moda':
                self._log("Tratamento de Nulos: Preenchendo com a MODA das colunas...")
                return X.fillna(X.mode().iloc[0])
            case '0':
                self._log("Tratamento de Nulos: Preenchendo com ZERO.")
                return X.fillna(0)
            case _:
                self._log(f"Aviso: Método de preenchimento '{method}' desconhecido. Usando ZERO por padrão.")
                return X.fillna(0)

    def _encode_labels(self, y_series, binary_label):
        y_str = y_series.astype(str).str.strip()
        
        if binary_label:
            self._log("Target: Binarizando rótulos (0=BENIGN, 1=ATTACK)...")
            is_benign = y_str.str.upper() == 'BENIGN'
            y = np.where(is_benign, 0, 1).astype(np.int8)
            target_names = ['BENIGN', 'ATTACK'] 
        else:
            self._log("Target: Mantendo multiclasse (Forçando BENIGN=0)...")
            unique_labels = y_str.unique().tolist()
            
            # Encontra o label normal (BENIGN) e força ele a ser o índice 0
            normal_label = next((l for l in unique_labels if l.upper() in ['BENIGN', 'NORMAL']), None)
            if normal_label and normal_label in unique_labels:
                unique_labels.remove(normal_label)
                unique_labels.insert(0, normal_label) # Coloca na posição 0
            
            # Mapeia as strings para inteiros respeitando a nova ordem
            mapping = {label: idx for idx, label in enumerate(unique_labels)}
            y = y_str.map(mapping).fillna(-1).astype(np.int8)
            target_names = unique_labels

        return y, target_names

    def create_stream(self, df, target_label_col='Label', binary_label=True, 
                    normalize_method=None, threshold_var=None,
                    threshold_corr=None, top_n_features=None,
                    return_stream=True, extra_ignore_cols=None,
                    imputation_method='0'):

        # limpeza básica
        self._log("Limpeza: Removendo espaços, identificadores e colunas vazias...")
        df.columns = df.columns.str.strip()
        target_label_col = target_label_col.strip()
        
        # Aplicação do filtro global de features antes de qualquer processamento
        if self.selected_features is not None:
            self._log(f"Filtro Global Ativo: Mantendo apenas as {len(self.selected_features)} features especificadas.")
            
            # Cria a lista do que manter, garantindo que as features existem no DF
            cols_to_keep = [c for c in self.selected_features if c in df.columns]
            
            # Trava de segurança: Garante que a coluna target NÃO seja excluída
            if target_label_col in df.columns and target_label_col not in cols_to_keep:
                cols_to_keep.append(target_label_col)
                
            df = df[cols_to_keep]

        ignore_cols = ['Flow ID', 'Timestamp', 'SimillarHTTP', 'Unnamed: 0']
        if extra_ignore_cols:
            if isinstance(extra_ignore_cols, str):
                ignore_cols.append(extra_ignore_cols)
            else:
                ignore_cols.extend(extra_ignore_cols)
                
        cols_to_drop = [c for c in ignore_cols if c in df.columns]
        X = df.drop(columns=[target_label_col] + cols_to_drop, errors='ignore')
        
        # tratamento numérico
        self._log("Pré-processamento: Convertendo infinitos...")
        X = X.select_dtypes(include=[np.number])
        X.replace([np.inf, -np.inf], [np.finfo(np.float32).max, np.finfo(np.float32).min], inplace=True)
        X = self._handle_missing_values(X, method=imputation_method)

        # normalização 
        if normalize_method:
            temp_col_names = X.columns
            temp_x_array = self._normalize_data(X, method=normalize_method)
            X = pd.DataFrame(temp_x_array, columns=temp_col_names)

        # Definição do target e encoding
        y, target_names = self._encode_labels(df[target_label_col], binary_label)

        # redução da dimensionalidade (agora atua apenas sobre as features já filtradas)
        if threshold_var is not None or threshold_corr is not None or top_n_features is not None:
            self._log("Seleção de Features: Iniciando pipeline de redução de dimensionalidade...")
            X = self._remove_features(X, y, threshold_var=threshold_var,
                                    threshold_corr=threshold_corr, top_n_features=top_n_features)
        else:
            self._log("Seleção de Features: Nenhuma técnica dinâmica selecionada. Mantendo colunas atuais.")

        # extrai dados finais para retorno 
        feature_names = X.columns.tolist()
        final_x_array = X.values

        # criação do retorno
        if return_stream:
            self._log("Finalização: Criando objeto NumpyStream para o CapyMOA.\n")
            stream_obj = NumpyStream(
                final_x_array, y, target_name="Class", 
                feature_names=feature_names, target_type="categorical"
            )
            return stream_obj, target_names, feature_names
        else:
            self._log("Finalização: Retornando DataFrame pandas processado.\n")
            final_df = pd.DataFrame(final_x_array, columns=feature_names)
            return final_df, y, target_names