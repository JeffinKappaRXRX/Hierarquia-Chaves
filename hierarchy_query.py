import pandas as pd
import streamlit as st
from collections import defaultdict
from typing import List, Dict

class HierarchyQuery:
    """
    Classe para consultas rápidas de hierarquia usando formato Name -> Name Bloco.
    """
    
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()
        self.jusante_map = defaultdict(list)  # pai -> [filhos]
        self.montante_map = {}  # filho -> pai
        self._build_maps()
    
    def _build_maps(self):
        """Constrói os mapas de relação jusante e montante."""
        # Normaliza os dados
        self.df['Name'] = self.df['Name'].astype(str).str.strip()
        self.df['Name Bloco'] = self.df['Name Bloco'].astype(str).str.strip()
        
        # Constrói os mapas
        for _, row in self.df.iterrows():
            filho = str(row['Name']).strip()
            pai = str(row['Name Bloco']).strip()
            
            # Verifica se pai é válido
            if pai and pai not in ['nan', 'None', '', 'NULL', 'NaN']:
                # Jusante: pai -> [filhos]
                if filho not in self.jusante_map[pai]:
                    self.jusante_map[pai].append(filho)
                
                # Montante: filho -> pai
                self.montante_map[filho] = pai
    
    def get_jusante(self, chave: str) -> List[str]:
        """
        Obtém todas as chaves a jusante de uma chave (recursivo).
        
        Args:
            chave: Nome da chave
            
        Returns:
            Lista de todas as chaves descendentes
        """
        result = []
        filhos = self.jusante_map.get(chave, [])
        
        for filho in filhos:
            result.append(filho)
            result.extend(self.get_jusante(filho))
        
        return result
    
    def get_montante(self, chave: str) -> List[str]:
        """
        Obtém todas as chaves a montante de uma chave (recursivo).
        
        Args:
            chave: Nome da chave
            
        Returns:
            Lista de todas as chaves ancestrais
        """
        result = []
        pai = self.montante_map.get(chave)
        
        if pai:
            result.append(pai)
            result.extend(self.get_montante(pai))
        
        return result
    
    def get_filhos_diretos(self, chave: str) -> List[str]:
        """Obtém apenas os filhos diretos de uma chave."""
        return self.jusante_map.get(chave, [])
    
    def get_pai_direto(self, chave: str) -> str:
        """Obtém apenas o pai direto de uma chave."""
        return self.montante_map.get(chave, '')
    
    def get_info_chave(self, chave: str) -> Dict:
        """
        Obtém informações completas sobre uma chave.
        
        Args:
            chave: Nome da chave
            
        Returns:
            Dicionário com informações da chave
        """
        jusantes = self.get_jusante(chave)
        montantes = self.get_montante(chave)
        filhos_diretos = self.get_filhos_diretos(chave)
        pai_direto = self.get_pai_direto(chave)
        
        return {
            'chave': chave,
            'pai_direto': pai_direto,
            'filhos_diretos': filhos_diretos,
            'total_jusantes': len(jusantes),
            'total_montantes': len(montantes),
            'jusantes': jusantes,
            'montantes': montantes,
            'nivel_hierarquico': len(montantes) + 1,
            'is_raiz': len(montantes) == 0,
            'is_folha': len(jusantes) == 0
        }
    
    def search_chave(self, termo: str) -> List[str]:
        """
        Busca chaves que contêm o termo especificado.
        
        Args:
            termo: Termo de busca
            
        Returns:
            Lista de chaves que contêm o termo
        """
        todas_chaves = set(self.df['Name'].values)
        todas_chaves.update(self.montante_map.keys())
        todas_chaves.update(self.jusante_map.keys())
        
        return [chave for chave in todas_chaves 
                if termo.upper() in str(chave).upper()]
    
    def export_hierarquia(self, chave: str) -> Dict[str, pd.DataFrame]:
        """
        Exporta a hierarquia de uma chave para DataFrames.
        
        Args:
            chave: Nome da chave
            
        Returns:
            Dicionário com DataFrames de jusante e montante
        """
        info = self.get_info_chave(chave)
        
        # DataFrame jusante
        df_jusante = pd.DataFrame({
            'Chave_Jusante': info['jusantes']
        }) if info['jusantes'] else pd.DataFrame()
        
        # DataFrame montante
        df_montante = pd.DataFrame({
            'Chave_Montante': info['montantes']
        }) if info['montantes'] else pd.DataFrame()
        
        return {
            'jusante': df_jusante,
            'montante': df_montante,
            'info': pd.DataFrame([info])
        }