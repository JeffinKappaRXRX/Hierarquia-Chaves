import pandas as pd
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any

class HierarchyProcessor:
    """
    Classe para processar hierarquias de chaves elétricas a partir de relações chave-bloco.
    """
    
    def __init__(self, dataframe: pd.DataFrame):
        """
        Inicializa o processador com um DataFrame contendo dados de chaves.
        
        Args:
            dataframe: DataFrame com colunas 'Name' e 'Name Bloco'
        """
        self.df = dataframe.copy()
        self.tree = defaultdict(list)  # pai -> [filhos]
        self.reverse_tree = {}  # filho -> pai
        self.processed = False
        self.statistics = {}
        
    def build_tree_from_blocks(self) -> None:
        """
        Constrói uma árvore hierárquica a partir das relações Name -> Name Bloco.
        """
        self.tree = defaultdict(list)
        self.reverse_tree = {}
        
        # Processa cada linha do DataFrame
        for _, row in self.df.iterrows():
            chave_filho = str(row['Name']).strip()
            chave_pai = str(row['Name Bloco']).strip()
            
            # Ignora registros sem pai válido
            if chave_pai and chave_pai not in ['nan', 'None', '', 'NULL']:
                # Adiciona relação pai -> filho
                if chave_filho not in self.tree[chave_pai]:
                    self.tree[chave_pai].append(chave_filho)
                
                # Adiciona relação reversa filho -> pai
                self.reverse_tree[chave_filho] = chave_pai
    
    def get_downstream_keys(self, node: str, tree: Dict[str, List[str]]) -> List[str]:
        """
        Obtém as chaves diretamente a jusante (filhas) de um nó.
        
        Args:
            node: Nome da chave pai
            tree: Árvore de relações
            
        Returns:
            Lista de chaves filhas diretas
        """
        return tree.get(node, [])
    
    def get_all_downstream_keys(self, node: str, tree: Dict[str, List[str]], 
                               visited: Optional[Set[str]] = None) -> List[str]:
        """
        Obtém todas as chaves a jusante (incluindo netos, bisnetos, etc.) de um nó.
        
        Args:
            node: Nome da chave pai
            tree: Árvore de relações
            visited: Conjunto de nós já visitados para evitar loops
            
        Returns:
            Lista de todas as chaves descendentes
        """
        if visited is None:
            visited = set()
            
        if node in visited:
            return []
            
        visited.add(node)
        all_downstream = []
        
        direct_children = tree.get(node, [])
        all_downstream.extend(direct_children)
        
        # Recursivamente busca descendentes
        for child in direct_children:
            all_downstream.extend(
                self.get_all_downstream_keys(child, tree, visited.copy())
            )
        
        return all_downstream
    
    def get_upstream_chain(self, node: str) -> List[str]:
        """
        Obtém toda a cadeia hierárquica ascendente de um nó (filho → pai → avô → bisavô).
        
        Args:
            node: Nome da chave para encontrar a cadeia ascendente
            
        Returns:
            Lista ordenada da cadeia hierárquica (do mais específico ao mais geral)
        """
        chain = []
        current_node = node
        visited = set()  # Para evitar loops infinitos
        
        # Navega pela cadeia hierárquica usando reverse_tree
        while current_node in self.reverse_tree and current_node not in visited:
            visited.add(current_node)
            parent = self.reverse_tree[current_node]
            chain.append(parent)
            current_node = parent
            
        return chain
    
    def get_full_hierarchy_path(self, node: str) -> str:
        """
        Retorna o caminho hierárquico completo de um nó.
        
        Args:
            node: Nome da chave
            
        Returns:
            String com caminho completo (ex: "Subestacao\\Alimentador\\Chave_Principal\\Chave_Atual")
        """
        upstream_chain = self.get_upstream_chain(node)
        
        if not upstream_chain:
            return node
            
        # Inverte a ordem para ter da raiz até o nó atual
        full_path = upstream_chain[::-1] + [node]
        return '\\'.join(full_path)
    
    def get_all_related_keys(self, node: str) -> Dict[str, Any]:
        """
        Obtém todas as chaves relacionadas a um nó (ancestrais e descendentes).
        
        Args:
            node: Nome da chave central
            
        Returns:
            Dicionário com ancestrais, descendentes diretos e todos os descendentes
        """
        return {
            'ancestrais': self.get_upstream_chain(node),
            'filhos_diretos': self.get_downstream_keys(node, self.tree),
            'todos_descendentes': self.get_all_downstream_keys(node, self.tree),
            'caminho_completo': self.get_full_hierarchy_path(node)
        }
    
    def build_reverse_tree(self) -> Dict[str, str]:
        """
        Constrói árvore inversa para navegação rápida (filho -> pai).
        
        Returns:
            Dicionário mapeando cada filho ao seu pai
        """
        reverse_tree = {}
        for parent, children in self.tree.items():
            for child in children:
                reverse_tree[child] = parent
        return reverse_tree
    
    def get_path_depth(self, path: str) -> int:
        """
        Calcula a profundidade de um caminho hierárquico.
        
        Args:
            path: Caminho no formato \Chave_1\Chave_2\Chave_3
            
        Returns:
            Profundidade do caminho (número de níveis)
        """
        if pd.isna(path) or not isinstance(path, str):
            return 0
        
        parts = [part.strip() for part in path.split('\\') if part.strip()]
        return len(parts)
    
    def get_root_nodes(self) -> List[str]:
        """
        Identifica os nós raiz da árvore (nós que não são filhos de nenhum outro).
        
        Returns:
            Lista de nós raiz
        """
        all_children = set()
        for children in self.tree.values():
            all_children.update(children)
        
        all_parents = set(self.tree.keys())
        root_nodes = all_parents - all_children
        
        return sorted(list(root_nodes))
    
    def get_leaf_nodes(self) -> List[str]:
        """
        Identifica os nós folha da árvore (nós que não têm filhos).
        
        Returns:
            Lista de nós folha
        """
        all_children = set()
        for children in self.tree.values():
            all_children.update(children)
        
        leaf_nodes = []
        for node in all_children:
            if node not in self.tree or not self.tree[node]:
                leaf_nodes.append(node)
        
        return sorted(leaf_nodes)
    
    def calculate_statistics(self) -> Dict:
        """
        Calcula estatísticas da árvore hierárquica.
        
        Returns:
            Dicionário com estatísticas da árvore
        """
        if not self.processed:
            return {}
        
        # Estatísticas básicas
        total_nodes = len(set(self.tree.keys()) | 
                         set([child for children in self.tree.values() for child in children]))
        total_relationships = sum(len(children) for children in self.tree.values())
        root_nodes = self.get_root_nodes()
        leaf_nodes = self.get_leaf_nodes()
        
        # Profundidade máxima
        max_depth = 0
        if not self.df['Caminho completo'].empty:
            max_depth = self.df['Caminho completo'].apply(self.get_path_depth).max()
        
        # Distribuição de filhos
        children_counts = [len(children) for children in self.tree.values()]
        avg_children = sum(children_counts) / len(children_counts) if children_counts else 0
        max_children = max(children_counts) if children_counts else 0
        
        self.statistics = {
            'total_nodes': int(total_nodes),
            'total_relationships': int(total_relationships),
            'root_nodes_count': len(root_nodes),
            'leaf_nodes_count': len(leaf_nodes),
            'max_depth': int(max_depth),
            'avg_children_per_node': round(avg_children, 2),
            'max_children_per_node': int(max_children),
            'tree_density': round(total_relationships / total_nodes if total_nodes > 0 else 0, 2)
        }
        
        return self.statistics
    
    def process(self) -> pd.DataFrame:
        """
        Processa o DataFrame para adicionar informações de hierarquia completa.
        
        Returns:
            DataFrame processado com colunas de hierarquia adicionadas
        """
        # Detecta qual formato está sendo usado
        if 'Name Bloco' in self.df.columns:
            return self._process_from_name_bloco()
        elif 'Caminho completo' in self.df.columns:
            return self._process_from_caminho_completo()
        else:
            raise ValueError("O arquivo deve conter 'Name Bloco' ou 'Caminho completo'")
    
    def _process_from_name_bloco(self) -> pd.DataFrame:
        """
        Processa DataFrame usando formato Name -> Name Bloco.
        """
        from collections import defaultdict
        
        df = self.df.copy()
        df['Name'] = df['Name'].astype(str).str.strip()
        df['Name Bloco'] = df['Name Bloco'].astype(str).str.strip()
        
        # Mapeamento de pais e filhos
        filhos_map = defaultdict(list)
        pais_map = {}
        
        for _, row in df.iterrows():
            filho = row['Name']
            pai = row['Name Bloco']
            if pai and pai not in ['nan', 'None', '', 'NULL', 'NaN']:
                filhos_map[pai].append(filho)
                pais_map[filho] = pai
        
        # Funções recursivas
        def buscar_montante(chave):
            if chave in pais_map:
                pai = pais_map[chave]
                return [pai] + buscar_montante(pai)
            return []
        
        def buscar_jusante(chave):
            filhos = filhos_map.get(chave, [])
            todos = []
            for f in filhos:
                todos.append(f)
                todos.extend(buscar_jusante(f))
            return todos
        
        # Adicionar colunas novas
        df['chaves-jusante'] = df['Name'].map(lambda x: ', '.join(filhos_map.get(x, [])))
        df['chaves-montante'] = df['Name'].map(lambda x: ', '.join(buscar_montante(x)))
        df['todas-chaves-jusante'] = df['Name'].map(lambda x: ', '.join(buscar_jusante(x)))
        df['caminho-hierarquico'] = df['Name'].map(lambda x: '\\' + '\\'.join(reversed(buscar_montante(x) + [x])) if buscar_montante(x) else x)
        df['nivel-hierarquico'] = df['chaves-montante'].apply(lambda x: len(x.split(', ')) if x else 1)
        df['is-raiz'] = df['Name'].apply(lambda x: x not in pais_map)
        df['is-folha'] = df['Name'].apply(lambda x: len(filhos_map.get(x, [])) == 0)
        
        # Atualiza as estruturas internas para compatibilidade
        self.tree = dict(filhos_map)
        self.reverse_tree = pais_map
        self.processed = True
        
        return df
    
    def _process_from_caminho_completo(self) -> pd.DataFrame:
        """
        Processa DataFrame usando formato Caminho completo (método original).
        """
        # Constrói a árvore a partir das relações chave-bloco
        self.build_tree_from_blocks()
        
        # Processa cada linha para encontrar todas as relações hierárquicas
        def get_downstream_for_name(name):
            downstream = self.get_downstream_keys(name, self.tree)
            return ', '.join(downstream) if downstream else ''
        
        def get_all_downstream_for_name(name):
            all_downstream = self.get_all_downstream_keys(name, self.tree)
            return ', '.join(all_downstream) if all_downstream else ''
        
        def get_upstream_for_name(name):
            upstream = self.get_upstream_chain(name)
            return ', '.join(upstream) if upstream else ''
        
        def get_hierarchy_path_for_name(name):
            return self.get_full_hierarchy_path(name)
        
        def get_hierarchy_level(name):
            upstream = self.get_upstream_chain(name)
            return len(upstream) + 1  # +1 para incluir o próprio nó
        
        def get_parent_key(name):
            return self.reverse_tree.get(name, '')
        
        def is_root_node(name):
            return name in self.get_root_nodes()
        
        def is_leaf_node(name):
            return name in self.get_leaf_nodes()
        
        # Adiciona colunas com informações hierárquicas
        self.df['chave-carga'] = self.df['Name'].apply(get_downstream_for_name)
        self.df['todas-chaves-jusante'] = self.df['Name'].apply(get_all_downstream_for_name)
        self.df['chaves-montante'] = self.df['Name'].apply(get_upstream_for_name)
        self.df['caminho-hierarquico'] = self.df['Name'].apply(get_hierarchy_path_for_name)
        self.df['nivel-hierarquico'] = self.df['Name'].apply(get_hierarchy_level)
        self.df['chave-pai'] = self.df['Name'].apply(get_parent_key)
        self.df['is-raiz'] = self.df['Name'].apply(is_root_node)
        self.df['is-folha'] = self.df['Name'].apply(is_leaf_node)
        
        # Calcula estatísticas
        self.calculate_statistics()
        self.processed = True
        
        return self.df
    
    def get_statistics(self) -> Dict:
        """
        Retorna as estatísticas da árvore.
        
        Returns:
            Dicionário com estatísticas
        """
        if not self.processed:
            return {}
        return self.statistics
    
    def get_node_info(self, node_name: str) -> Dict:
        """
        Obtém informações detalhadas sobre um nó específico.
        
        Args:
            node_name: Nome do nó
            
        Returns:
            Dicionário com informações do nó
        """
        if not self.processed:
            return {}
        
        info = {
            'name': node_name,
            'direct_children': self.get_downstream_keys(node_name, self.tree),
            'total_descendants': len(self.get_all_downstream_keys(node_name, self.tree)),
            'is_root': node_name in self.get_root_nodes(),
            'is_leaf': node_name in self.get_leaf_nodes()
        }
        
        # Encontra o pai do nó
        parent = None
        for potential_parent, children in self.tree.items():
            if node_name in children:
                parent = potential_parent
                break
        info['parent'] = parent
        
        return info
