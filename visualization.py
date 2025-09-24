import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import networkx as nx
import math

class HierarchyVisualizer:
    """
    Classe para visualizar hierarquias de chaves elétricas usando Plotly.
    """
    
    def __init__(self, tree: Dict[str, List[str]]):
        """
        Inicializa o visualizador com uma árvore de hierarquia.
        
        Args:
            tree: Dicionário representando a árvore hierárquica
        """
        self.tree = tree
        self.colors = px.colors.qualitative.Set3
    
    def _get_all_nodes(self) -> List[str]:
        """
        Obtém todos os nós da árvore.
        
        Returns:
            Lista de todos os nós únicos
        """
        all_nodes = set(self.tree.keys())
        for children in self.tree.values():
            all_nodes.update(children)
        return list(all_nodes)
    
    def _get_root_nodes(self) -> List[str]:
        """
        Identifica os nós raiz da árvore.
        
        Returns:
            Lista de nós raiz
        """
        all_children = set()
        for children in self.tree.values():
            all_children.update(children)
        
        all_parents = set(self.tree.keys())
        return list(all_parents - all_children)
    
    def _calculate_positions_hierarchical(self, max_depth: int = 5) -> Tuple[Dict[str, Tuple[float, float]], List[Tuple[str, str]]]:
        """
        Calcula posições hierárquicas para os nós usando layout em árvore.
        
        Args:
            max_depth: Profundidade máxima para visualização
            
        Returns:
            Tupla contendo posições dos nós e lista de arestas
        """
        positions = {}
        edges = []
        root_nodes = self._get_root_nodes()
        
        if not root_nodes:
            return {}, []
        
        # Configurações de layout
        level_height = 1.0
        base_width = 2.0
        
        # Processa cada árvore raiz separadamente
        current_x_offset = 0
        
        for root in root_nodes:
            # BFS para calcular posições
            queue = deque([(root, 0, current_x_offset)])  # (nó, nível, x_offset)
            level_nodes = defaultdict(list)
            visited = set()
            
            while queue:
                node, level, x_offset = queue.popleft()
                
                if node in visited or level >= max_depth:
                    continue
                    
                visited.add(node)
                level_nodes[level].append((node, x_offset))
                
                # Adiciona filhos à fila
                children = self.tree.get(node, [])
                child_spacing = base_width / (len(children) + 1) if children else 0
                
                for i, child in enumerate(children):
                    child_x = x_offset + (i + 1) * child_spacing
                    queue.append((child, level + 1, child_x))
                    edges.append((node, child))
            
            # Calcula posições finais para esta subárvore
            for level, nodes in level_nodes.items():
                y = -level * level_height
                
                # Centraliza nós no nível
                if len(nodes) == 1:
                    node, x_offset = nodes[0]
                    positions[node] = (current_x_offset, y)
                else:
                    level_width = base_width * 2
                    for i, (node, _) in enumerate(nodes):
                        x = current_x_offset + (i - len(nodes)/2 + 0.5) * (level_width / len(nodes))
                        positions[node] = (x, y)
            
            # Atualiza offset para próxima árvore
            current_x_offset += base_width * 3
        
        return positions, edges
    
    def _calculate_positions_circular(self) -> Tuple[Dict[str, Tuple[float, float]], List[Tuple[str, str]]]:
        """
        Calcula posições em layout circular para os nós.
        
        Returns:
            Tupla contendo posições dos nós e lista de arestas
        """
        all_nodes = self._get_all_nodes()
        if not all_nodes:
            return {}, []
        
        positions = {}
        edges = []
        
        # Layout circular para nós
        n_nodes = len(all_nodes)
        for i, node in enumerate(all_nodes):
            angle = 2 * math.pi * i / n_nodes
            x = math.cos(angle)
            y = math.sin(angle)
            positions[node] = (x, y)
        
        # Adiciona arestas
        for parent, children in self.tree.items():
            for child in children:
                edges.append((parent, child))
        
        return positions, edges
    
    def create_tree_plot(self, max_depth: int = 5, layout: str = 'hierarchical') -> Optional[go.Figure]:
        """
        Cria um gráfico de árvore hierárquica.
        
        Args:
            max_depth: Profundidade máxima para visualização
            layout: Tipo de layout ('hierarchical' ou 'circular')
            
        Returns:
            Figura Plotly ou None se não houver dados
        """
        if not self.tree:
            return None
        
        # Calcula posições
        if layout == 'circular':
            positions, edges = self._calculate_positions_circular()
        else:
            positions, edges = self._calculate_positions_hierarchical(max_depth)
        
        if not positions:
            return None
        
        # Cria figura
        fig = go.Figure()
        
        # Adiciona arestas
        edge_x = []
        edge_y = []
        for parent, child in edges:
            if parent in positions and child in positions:
                x0, y0 = positions[parent]
                x1, y1 = positions[child]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
        
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='rgba(125, 125, 125, 0.5)'),
            hoverinfo='none',
            mode='lines',
            name='Conexões'
        ))
        
        # Adiciona nós
        node_x = []
        node_y = []
        node_text = []
        node_info = []
        
        for node, (x, y) in positions.items():
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            
            # Informações do hover
            children = self.tree.get(node, [])
            info = f"<b>{node}</b><br>"
            if children:
                info += f"Filhas: {', '.join(children[:5])}"
                if len(children) > 5:
                    info += f"<br>... e mais {len(children) - 5}"
            else:
                info += "Nó folha"
            node_info.append(info)
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="middle center",
            textfont=dict(size=10),
            hovertemplate='%{hovertext}<extra></extra>',
            hovertext=node_info,
            marker=dict(
                size=20,
                color='lightblue',
                line=dict(width=2, color='darkblue')
            ),
            name='Chaves'
        ))
        
        fig.update_layout(
            title=f"Hierarquia de Chaves Elétricas - Layout {layout.title()}",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Clique nos nós para ver detalhes",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(color="gray", size=12)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(0,0,0,0)',
            height=600
        )
        
        return fig
    
    def create_subtree_plot(self, root_node: str, max_depth: int = 5) -> Optional[go.Figure]:
        """
        Cria um gráfico de subárvore a partir de um nó raiz específico.
        
        Args:
            root_node: Nó raiz da subárvore
            max_depth: Profundidade máxima para visualização
            
        Returns:
            Figura Plotly ou None se não houver dados
        """
        if root_node not in self.tree and not any(root_node in children for children in self.tree.values()):
            return None
        
        # Cria subárvore
        subtree = {}
        visited = set()
        queue = deque([(root_node, 0)])
        
        while queue:
            node, depth = queue.popleft()
            
            if node in visited or depth >= max_depth:
                continue
                
            visited.add(node)
            children = self.tree.get(node, [])
            subtree[node] = children
            
            for child in children:
                queue.append((child, depth + 1))
        
        # Cria visualizador temporário para subárvore
        temp_visualizer = HierarchyVisualizer(subtree)
        fig = temp_visualizer.create_tree_plot(max_depth, 'hierarchical')
        
        if fig:
            fig.update_layout(
                title=f"Subárvore: {root_node} (Profundidade: {max_depth})"
            )
        
        return fig
    
    def create_network_graph(self) -> Optional[go.Figure]:
        """
        Cria um gráfico de rede usando NetworkX para layout automático.
        
        Returns:
            Figura Plotly ou None se não houver dados
        """
        if not self.tree:
            return None
        
        try:
            # Cria grafo NetworkX
            G = nx.DiGraph()
            
            for parent, children in self.tree.items():
                for child in children:
                    G.add_edge(parent, child)
            
            if not G.nodes():
                return None
            
            # Calcula layout
            pos = nx.spring_layout(G, k=3, iterations=50)
            
            # Prepara dados para Plotly
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            node_x = []
            node_y = []
            node_text = []
            node_info = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
                
                # Calcula grau de entrada e saída
                in_degree = G.in_degree(node)
                out_degree = G.out_degree(node)
                info = f"<b>{node}</b><br>Entrada: {in_degree}<br>Saída: {out_degree}"
                node_info.append(info)
            
            # Cria figura
            fig = go.Figure()
            
            # Adiciona arestas
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, color='gray'),
                hoverinfo='none',
                mode='lines',
                name='Conexões'
            ))
            
            # Adiciona nós
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=node_text,
                textposition="middle center",
                textfont=dict(size=8),
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=node_info,
                marker=dict(
                    size=15,
                    color='lightcoral',
                    line=dict(width=1, color='darkred')
                ),
                name='Chaves'
            ))
            
            fig.update_layout(
                title="Rede de Chaves Elétricas (Layout Automático)",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='rgba(0,0,0,0)',
                height=600
            )
            
            return fig
            
        except Exception:
            # Se NetworkX falhar, retorna visualização básica
            return self.create_tree_plot()
    
    def create_statistics_charts(self, statistics: Dict) -> go.Figure:
        """
        Cria gráficos com estatísticas da hierarquia.
        
        Args:
            statistics: Dicionário com estatísticas
            
        Returns:
            Figura Plotly com múltiplos gráficos
        """
        if not statistics:
            return go.Figure()
        
        # Cria subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Distribuição de Nós',
                'Métricas da Árvore',
                'Profundidade e Largura',
                'Densidade da Rede'
            ],
            specs=[[{"type": "bar"}, {"type": "indicator"}],
                   [{"type": "bar"}, {"type": "indicator"}]]
        )
        
        # Gráfico 1: Distribuição de nós
        node_types = ['Raiz', 'Folha', 'Intermediários']
        node_counts = [
            statistics.get('root_nodes_count', 0),
            statistics.get('leaf_nodes_count', 0),
            statistics.get('total_nodes', 0) - statistics.get('root_nodes_count', 0) - statistics.get('leaf_nodes_count', 0)
        ]
        
        fig.add_trace(
            go.Bar(x=node_types, y=node_counts, marker_color=['red', 'green', 'blue']),
            row=1, col=1
        )
        
        # Gráfico 2: Métrica de relacionamentos
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=statistics.get('total_relationships', 0),
                title={'text': "Total de Relacionamentos"},
                gauge={'axis': {'range': [None, statistics.get('total_relationships', 0) * 1.2]}}
            ),
            row=1, col=2
        )
        
        # Gráfico 3: Profundidade e filhos
        metrics = ['Profundidade Máx', 'Filhos Máx', 'Filhos Médio']
        values = [
            statistics.get('max_depth', 0),
            statistics.get('max_children_per_node', 0),
            statistics.get('avg_children_per_node', 0)
        ]
        
        fig.add_trace(
            go.Bar(x=metrics, y=values, marker_color=['orange', 'purple', 'cyan']),
            row=2, col=1
        )
        
        # Gráfico 4: Densidade
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=statistics.get('tree_density', 0),
                title={'text': "Densidade da Árvore"},
                delta={'reference': 1.0},
                gauge={
                    'axis': {'range': [None, 2.0]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.5], 'color': "lightgray"},
                        {'range': [0.5, 1.0], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 1.0
                    }
                }
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Estatísticas da Hierarquia de Chaves",
            showlegend=False,
            height=600
        )
        
        return fig
