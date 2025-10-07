"""
Fidelity-Aware Shortest Path (FASP) Router
Task T2.2 - Member 2

Implements Dijkstra's algorithm with fidelity-aware cost metric.
Integrates with SeQUeNCe NetworkManager for entanglement routing.
"""

import heapq
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class FASP_Router:
    """
    Fidelity-Aware Shortest Path Router for Quantum Networks.
    
    Uses Dijkstra's algorithm with logarithmic fidelity cost to find
    paths that maximize end-to-end entanglement quality.
    """
    
    def __init__(self, network, fidelity_estimator):
        """
        Initialize the FASP Router.
        
        Args:
            network: SeQUeNCe Network object
            fidelity_estimator: FidelityEstimator instance for cost calculations
        """
        self.network = network
        self.fidelity_estimator = fidelity_estimator
        self.graph = {}  # Adjacency list representation
        self.routing_table = {}  # Cache computed paths
        
        # Build network graph
        self._build_graph()
        
        logger.info("FASP_Router initialized with fidelity-aware routing")
    
    def _build_graph(self):
        """
        Build adjacency list representation of the network.
        Each edge is weighted by the fidelity cost: -log(F_l)
        """
        self.graph = defaultdict(list)
        
        # Iterate through all nodes
        for node_name, node in self.network.nodes.items():
            # Find all quantum channels (outgoing links)
            if hasattr(node, 'qchannels'):
                for qc_name, qc in node.qchannels.items():
                    if hasattr(qc, 'receiver') and hasattr(qc.receiver, 'owner'):
                        neighbor = qc.receiver.owner.name
                        
                        # Calculate fidelity-based cost for this link
                        cost = self.fidelity_estimator.get_link_cost(node_name, neighbor)
                        
                        # Add edge to graph
                        self.graph[node_name].append((neighbor, cost))
        
        logger.info(f"Built graph with {len(self.graph)} nodes")
        
        # Log graph structure for debugging
        for node, edges in self.graph.items():
            logger.debug(f"Node {node}: {len(edges)} neighbors")
    
    def find_path(self, source: str, destination: str) -> Optional[List[str]]:
        """
        Find the fidelity-aware shortest path using Dijkstra's algorithm.
        
        Args:
            source: Source node name
            destination: Destination node name
            
        Returns:
            List of node names forming the path, or None if no path exists
        """
        # Check cache
        path_key = (source, destination)
        if path_key in self.routing_table:
            logger.debug(f"Using cached path for {source} -> {destination}")
            return self.routing_table[path_key]
        
        # Dijkstra's algorithm with fidelity-aware costs
        path = self._dijkstra(source, destination)
        
        # Cache the result
        if path:
            self.routing_table[path_key] = path
            logger.info(f"Found path {source} -> {destination}: {' -> '.join(path)}")
        else:
            logger.warning(f"No path found from {source} to {destination}")
        
        return path
    
    def _dijkstra(self, source: str, destination: str) -> Optional[List[str]]:
        """
        Dijkstra's shortest path algorithm with fidelity costs.
        
        Args:
            source: Source node name
            destination: Destination node name
            
        Returns:
            Path as list of node names, or None if no path exists
        """
        # Initialize distances and predecessors
        distances = {node: float('inf') for node in self.graph.keys()}
        distances[source] = 0
        predecessors = {}
        
        # Priority queue: (cost, node)
        pq = [(0, source)]
        visited = set()
        
        while pq:
            current_cost, current_node = heapq.heappop(pq)
            
            # Skip if already visited
            if current_node in visited:
                continue
            
            visited.add(current_node)
            
            # Found destination
            if current_node == destination:
                return self._reconstruct_path(predecessors, source, destination)
            
            # Explore neighbors
            if current_node in self.graph:
                for neighbor, edge_cost in self.graph[current_node]:
                    if neighbor not in visited:
                        new_cost = current_cost + edge_cost
                        
                        # Update if better path found
                        if new_cost < distances[neighbor]:
                            distances[neighbor] = new_cost
                            predecessors[neighbor] = current_node
                            heapq.heappush(pq, (new_cost, neighbor))
        
        # No path found
        return None
    
    def _reconstruct_path(self, 
                         predecessors: Dict[str, str], 
                         source: str, 
                         destination: str) -> List[str]:
        """
        Reconstruct path from predecessors dictionary.
        
        Args:
            predecessors: Dictionary mapping nodes to their predecessors
            source: Source node
            destination: Destination node
            
        Returns:
            Path as list of node names
        """
        path = []
        current = destination
        
        while current != source:
            path.append(current)
            if current not in predecessors:
                logger.error(f"Path reconstruction failed at node {current}")
                return None
            current = predecessors[current]
        
        path.append(source)
        path.reverse()
        
        return path
    
    def get_path_metrics(self, path: List[str]) -> Dict[str, float]:
        """
        Calculate detailed metrics for a given path.
        
        Args:
            path: List of node names
            
        Returns:
            Dictionary with path metrics
        """
        if len(path) < 2:
            return {
                'fidelity': 1.0,
                'cost': 0.0,
                'hops': 0,
                'avg_link_fidelity': 1.0
            }
        
        # Calculate path fidelity
        path_fidelity = self.fidelity_estimator.calculate_path_fidelity(path)
        
        # Calculate total cost
        total_cost = 0
        link_fidelities = []
        
        for i in range(len(path) - 1):
            link_cost = self.fidelity_estimator.get_link_cost(path[i], path[i+1])
            total_cost += link_cost
            
            link_fid = self.fidelity_estimator.calculate_link_fidelity(path[i], path[i+1])
            link_fidelities.append(link_fid)
        
        return {
            'fidelity': path_fidelity,
            'cost': total_cost,
            'hops': len(path) - 1,
            'avg_link_fidelity': sum(link_fidelities) / len(link_fidelities),
            'min_link_fidelity': min(link_fidelities),
            'max_link_fidelity': max(link_fidelities)
        }
    
    def find_k_paths(self, source: str, destination: str, k: int = 3) -> List[List[str]]:
        """
        Find k-shortest paths using Yen's algorithm (simplified).
        Useful for comparing multiple routing options.
        
        Args:
            source: Source node name
            destination: Destination node name
            k: Number of paths to find
            
        Returns:
            List of paths (each path is a list of node names)
        """
        paths = []
        
        # Find first shortest path
        shortest_path = self.find_path(source, destination)
        if shortest_path:
            paths.append(shortest_path)
        else:
            return paths
        
        # For simplicity, we'll use a basic approach:
        # Remove edges from shortest path and find alternatives
        # (Full Yen's algorithm is more complex)
        
        candidate_paths = []
        
        for i in range(len(shortest_path) - 1):
            # Temporarily remove edge
            spur_node = shortest_path[i]
            root_path = shortest_path[:i+1]
            
            # Store original graph
            removed_edges = []
            
            # Remove edges that would create duplicate paths
            for path in paths:
                if len(path) > i and path[:i+1] == root_path:
                    if i+1 < len(path):
                        edge_to_remove = (path[i], path[i+1])
                        if spur_node in self.graph:
                            for j, (neighbor, cost) in enumerate(self.graph[spur_node]):
                                if neighbor == edge_to_remove[1]:
                                    removed_edges.append((spur_node, j, neighbor, cost))
            
            # Remove edges
            for node, idx, neighbor, cost in removed_edges:
                self.graph[node].pop(idx)
            
            # Find spur path
            spur_path = self._dijkstra(spur_node, destination)
            
            # Restore edges
            for node, idx, neighbor, cost in removed_edges:
                self.graph[node].insert(idx, (neighbor, cost))
            
            if spur_path:
                total_path = root_path[:-1] + spur_path
                if total_path not in paths:
                    candidate_paths.append(total_path)
        
        # Sort candidates by cost and add to paths
        candidate_paths.sort(key=lambda p: self.get_path_metrics(p)['cost'])
        
        for path in candidate_paths:
            if len(paths) >= k:
                break
            if path not in paths:
                paths.append(path)
        
        return paths[:k]
    
    def compare_with_baseline(self, source: str, destination: str) -> Dict:
        """
        Compare FASP path with traditional hop-count shortest path.
        
        Args:
            source: Source node name
            destination: Destination node name
            
        Returns:
            Dictionary with comparison metrics
        """
        # Get FASP path
        fasp_path = self.find_path(source, destination)
        fasp_metrics = self.get_path_metrics(fasp_path) if fasp_path else None
        
        # Get hop-count shortest path (temporarily use uniform costs)
        original_graph = self.graph.copy()
        
        # Build hop-count graph (all edges cost 1)
        self.graph = defaultdict(list)
        for node_name, node in self.network.nodes.items():
            if hasattr(node, 'qchannels'):
                for qc_name, qc in node.qchannels.items():
                    if hasattr(qc, 'receiver') and hasattr(qc.receiver, 'owner'):
                        neighbor = qc.receiver.owner.name
                        self.graph[node_name].append((neighbor, 1.0))
        
        baseline_path = self._dijkstra(source, destination)
        baseline_metrics = None
        if baseline_path:
            # Calculate metrics using actual fidelities
            baseline_metrics = {
                'fidelity': self.fidelity_estimator.calculate_path_fidelity(baseline_path),
                'hops': len(baseline_path) - 1,
                'path': baseline_path
            }
        
        # Restore original graph
        self.graph = original_graph
        
        comparison = {
            'fasp': {
                'path': fasp_path,
                'metrics': fasp_metrics
            },
            'baseline': {
                'path': baseline_path,
                'metrics': baseline_metrics
            }
        }
        
        # Calculate improvements
        if fasp_metrics and baseline_metrics:
            comparison['improvement'] = {
                'fidelity_gain': fasp_metrics['fidelity'] - baseline_metrics['fidelity'],
                'fidelity_gain_pct': (fasp_metrics['fidelity'] / baseline_metrics['fidelity'] - 1) * 100,
                'hop_difference': fasp_metrics['hops'] - baseline_metrics['hops']
            }
        
        return comparison
    
    def print_routing_table(self):
        """Print cached routing table for debugging."""
        print("\n" + "="*70)
        print("FASP ROUTING TABLE")
        print("="*70)
        print(f"{'Source':<10} {'Destination':<10} {'Path':<30} {'Hops':<5} {'Fidelity':<10}")
        print("-"*70)
        
        for (src, dst), path in sorted(self.routing_table.items()):
            metrics = self.get_path_metrics(path)
            path_str = ' -> '.join(path[:5])  # Show first 5 nodes
            if len(path) > 5:
                path_str += "..."
            print(f"{src:<10} {dst:<10} {path_str:<30} {metrics['hops']:<5} {metrics['fidelity']:.4f}")
        
        print("="*70 + "\n")
    
    def clear_cache(self):
        """Clear the routing table cache."""
        self.routing_table.clear()
        logger.info("Routing table cache cleared")


class BaselineRouter:
    """
    Traditional hop-count shortest path router for comparison.
    Uses uniform cost for all links (cost = 1 per hop).
    """
    
    def __init__(self, network, fidelity_estimator):
        """
        Initialize the Baseline Router.
        
        Args:
            network: SeQUeNCe Network object
            fidelity_estimator: FidelityEstimator for metrics (not used in routing)
        """
        self.network = network
        self.fidelity_estimator = fidelity_estimator
        self.graph = {}
        self.routing_table = {}
        
        self._build_graph()
        
        logger.info("BaselineRouter initialized with hop-count routing")
    
    def _build_graph(self):
        """Build graph with uniform edge costs (1 per hop)."""
        self.graph = defaultdict(list)
        
        for node_name, node in self.network.nodes.items():
            if hasattr(node, 'qchannels'):
                for qc_name, qc in node.qchannels.items():
                    if hasattr(qc, 'receiver') and hasattr(qc.receiver, 'owner'):
                        neighbor = qc.receiver.owner.name
                        # Uniform cost of 1 for hop-count routing
                        self.graph[node_name].append((neighbor, 1.0))
        
        logger.info(f"Baseline router graph built with {len(self.graph)} nodes")
    
    def find_path(self, source: str, destination: str) -> Optional[List[str]]:
        """
        Find shortest path by hop count (Dijkstra with uniform costs).
        
        Args:
            source: Source node name
            destination: Destination node name
            
        Returns:
            Path as list of node names
        """
        path_key = (source, destination)
        if path_key in self.routing_table:
            return self.routing_table[path_key]
        
        path = self._dijkstra(source, destination)
        
        if path:
            self.routing_table[path_key] = path
            logger.info(f"Baseline path {source} -> {destination}: {' -> '.join(path)}")
        
        return path
    
    def _dijkstra(self, source: str, destination: str) -> Optional[List[str]]:
        """Standard Dijkstra's algorithm with uniform costs."""
        distances = {node: float('inf') for node in self.graph.keys()}
        distances[source] = 0
        predecessors = {}
        
        pq = [(0, source)]
        visited = set()
        
        while pq:
            current_cost, current_node = heapq.heappop(pq)
            
            if current_node in visited:
                continue
            
            visited.add(current_node)
            
            if current_node == destination:
                return self._reconstruct_path(predecessors, source, destination)
            
            if current_node in self.graph:
                for neighbor, edge_cost in self.graph[current_node]:
                    if neighbor not in visited:
                        new_cost = current_cost + edge_cost
                        
                        if new_cost < distances[neighbor]:
                            distances[neighbor] = new_cost
                            predecessors[neighbor] = current_node
                            heapq.heappush(pq, (new_cost, neighbor))
        
        return None
    
    def _reconstruct_path(self, predecessors, source, destination):
        """Reconstruct path from predecessors."""
        path = []
        current = destination
        
        while current != source:
            path.append(current)
            if current not in predecessors:
                return None
            current = predecessors[current]
        
        path.append(source)
        path.reverse()
        return path
    
    def get_path_metrics(self, path: List[str]) -> Dict[str, float]:
        """Calculate metrics for a baseline path."""
        if len(path) < 2:
            return {'fidelity': 1.0, 'hops': 0}
        
        path_fidelity = self.fidelity_estimator.calculate_path_fidelity(path)
        
        return {
            'fidelity': path_fidelity,
            'hops': len(path) - 1
        }


# Test functions
if __name__ == "__main__":
    print("FASP_Router and BaselineRouter modules loaded successfully")
    print("These modules should be imported and used with SeQUeNCe Network objects")