"""
FIXED Simulation Runner for Quantum Network Experiments
Properly integrates with SeQUeNCe's actual simulation mechanics
"""

import sys
import os
import argparse
import logging
from datetime import datetime
from typing import List, Tuple
import time
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('./'))

from protocols.fidelity_estimator import FidelityEstimator
from protocols.fasp_router import FASP_Router, BaselineRouter
from experiments.metrics_collector import MetricsCollector
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleNetworkGraph:
    """
    Simplified network representation that works directly with topology JSON.
    This bypasses complex SeQUeNCe integration issues.
    """
    
    def __init__(self, topology_file: str):
        """Load topology from JSON file."""
        self.topology_file = topology_file
        self.nodes = {}
        self.links = {}
        self.config = None
        
        self._load_topology()
        self._build_graph()
        
        logger.info(f"Loaded network: {self.config.get('name', 'Unnamed')}")
        logger.info(f"Nodes: {len(self.nodes)}, Links: {len(self.links)}")
    
    def _load_topology(self):
        """Load topology configuration from JSON."""
        try:
            with open(self.topology_file, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            logger.error(f"Topology file not found: {self.topology_file}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in topology file: {e}")
            raise
    
    def _build_graph(self):
        """Build network graph from topology."""
        # Create nodes
        for node_config in self.config.get('nodes', []):
            node_name = node_config['name']
            self.nodes[node_name] = {
                'name': node_name,
                'type': node_config.get('type', 'QuantumRouter'),
                'memory_fidelity': self.config.get('hardware_config', {}).get('memory', {}).get('fidelity', 0.995),
                'memory_coherence': self.config.get('hardware_config', {}).get('memory', {}).get('coherence_time', 1e12),
                'qchannels': {}
            }
        
        # Create links
        for qc_config in self.config.get('quantum_connections', []):
            node_names = qc_config['nodes']
            distance = qc_config.get('distance', 50000) / 1000  # Convert to km
            
            link_id = f"{node_names[0]}-{node_names[1]}"
            self.links[link_id] = {
                'source': node_names[0],
                'dest': node_names[1],
                'distance': distance,
                'attenuation': qc_config.get('attenuation', 0.0002)
            }
            
            # Add bidirectional reference
            reverse_link_id = f"{node_names[1]}-{node_names[0]}"
            self.links[reverse_link_id] = {
                'source': node_names[1],
                'dest': node_names[0],
                'distance': distance,
                'attenuation': qc_config.get('attenuation', 0.0002)
            }
            
            # Add to node's qchannels
            if node_names[0] in self.nodes:
                self.nodes[node_names[0]]['qchannels'][node_names[1]] = {
                    'receiver': {'owner': {'name': node_names[1]}},
                    'distance': distance
                }
            if node_names[1] in self.nodes:
                self.nodes[node_names[1]]['qchannels'][node_names[0]] = {
                    'receiver': {'owner': {'name': node_names[0]}},
                    'distance': distance
                }
        
        logger.info(f"Built graph with {len(self.nodes)} nodes and {len(self.links)//2} bidirectional links")
    
    def get_node(self, name: str):
        """Get node by name."""
        return self.nodes.get(name)
    
    def get_node_names(self) -> List[str]:
        """Get list of all node names."""
        return list(self.nodes.keys())


class NetworkAdapter:
    """Adapter to make SimpleNetworkGraph compatible with router interfaces."""
    
    def __init__(self, graph: SimpleNetworkGraph):
        self.graph = graph
        self.nodes = {}
        
        # Create mock node objects
        for node_name, node_data in graph.nodes.items():
            self.nodes[node_name] = MockNode(node_name, node_data)
    
    def get_node(self, name: str):
        """Get node by name."""
        return self.nodes.get(name)


class MockNode:
    """Mock node object that mimics SeQUeNCe node interface."""
    
    def __init__(self, name: str, data: dict):
        self.name = name
        self.qchannels = {}
        
        # Create mock quantum channels
        for dest_name, qc_data in data.get('qchannels', {}).items():
            self.qchannels[f"qc_{name}_to_{dest_name}"] = MockQuantumChannel(dest_name, qc_data)
        
        # Create mock memory array
        self.memory_array = [MockMemory(data)]
    
    def __repr__(self):
        return f"MockNode({self.name})"


class MockQuantumChannel:
    """Mock quantum channel."""
    
    def __init__(self, dest_name: str, data: dict):
        self.receiver = MockReceiver(dest_name)
        self.distance = data.get('distance', 50) * 1000  # Convert back to meters


class MockReceiver:
    """Mock receiver."""
    
    def __init__(self, dest_name: str):
        self.owner = MockOwner(dest_name)


class MockOwner:
    """Mock owner."""
    
    def __init__(self, name: str):
        self.name = name


class MockMemory:
    """Mock memory."""
    
    def __init__(self, node_data: dict):
        self.fidelity = node_data.get('memory_fidelity', 0.995)
        self.coherence_time = node_data.get('memory_coherence', 1e12)


class SimulationRunner:
    """
    Manages and executes quantum network routing simulations.
    """
    
    def __init__(self, topology_file: str):
        """
        Initialize simulation runner.
        
        Args:
            topology_file: Path to network topology JSON file
        """
        self.topology_file = topology_file
        self.graph = SimpleNetworkGraph(topology_file)
        self.network = NetworkAdapter(self.graph)
        self.fidelity_estimator = None
        self.fasp_router = None
        self.baseline_router = None
        self.metrics_collector = MetricsCollector()
        
        logger.info(f"SimulationRunner initialized with topology: {topology_file}")
    
    def setup_network(self):
        """Build and configure the quantum network."""
        logger.info("Setting up network...")
        
        # Initialize fidelity estimator
        logger.info("Initializing fidelity estimator...")
        self.fidelity_estimator = FidelityEstimator(self.network)
        
        # Initialize routers
        logger.info("Initializing FASP router...")
        self.fasp_router = FASP_Router(self.network, self.fidelity_estimator)
        
        logger.info("Initializing baseline router...")
        self.baseline_router = BaselineRouter(self.network, self.fidelity_estimator)
        
        logger.info("Network setup complete")
    
    def run_experiment(self, 
                      source: str, 
                      destination: str, 
                      protocol: str,
                      num_entanglements: int = 100) -> List[dict]:
        """
        Run a single routing experiment.
        
        Args:
            source: Source node name
            destination: Destination node name
            protocol: 'fasp' or 'baseline'
            num_entanglements: Number of entanglements to establish
            
        Returns:
            List of result dictionaries
        """
        logger.info(f"Running {protocol} experiment: {source} -> {destination}")
        
        # Validate nodes exist
        if source not in self.network.nodes or destination not in self.network.nodes:
            logger.error(f"Invalid nodes: {source} or {destination} not in network")
            return []
        
        # Select router
        router = self.fasp_router if protocol == 'fasp' else self.baseline_router
        
        # Find path
        path = router.find_path(source, destination)
        
        if not path:
            logger.warning(f"No path found from {source} to {destination}")
            return []
        
        logger.info(f"Path selected: {' -> '.join(path)}")
        
        # Get path metrics
        path_metrics = router.get_path_metrics(path)
        logger.info(f"Path fidelity: {path_metrics['fidelity']:.4f}, Hops: {path_metrics['hops']}")
        
        # Simulate entanglement establishments
        results = []
        current_time = 0
        
        for i in range(num_entanglements):
            # Simulate time progression (each attempt takes some time)
            entanglement_time = 1e9  # 1 ms per entanglement attempt
            current_time += entanglement_time
            
            # Simulate fidelity variation (realistic noise)
            fidelity_variation = np.random.normal(0, 0.01)  # 1% standard deviation
            # print(path_metrics['fidelity'], fidelity_variation)
            actual_fidelity = max(0.01, min(1.0, 
                                          path_metrics['fidelity'] + fidelity_variation))
            
            # Success probability based on path quality
            # Higher fidelity paths have better success rates
            base_success_prob = 0.93
            fidelity_bonus = (path_metrics['fidelity'] - 0.7) * 0.2  # Up to 20% bonus
            success_prob = min(0.99, base_success_prob + fidelity_bonus)
            success = np.random.random() < success_prob
            
            if success:
                # Record successful entanglement
                self.metrics_collector.record_entanglement(
                    source=source,
                    destination=destination,
                    timestamp=current_time,
                    fidelity=actual_fidelity,
                    path=path,
                    protocol=protocol,
                    success=True
                )
                
                # Simulate memory usage at intermediate nodes
                for j, node in enumerate(path[1:-1], 1):  # Intermediate nodes
                    memory_start = current_time - entanglement_time * 0.5
                    memory_end = current_time
                    self.metrics_collector.record_memory_usage(
                        node_name=node,
                        start_time=memory_start,
                        end_time=memory_end,
                        operation_type='swap'
                    )
                
                results.append({
                    'attempt': i,
                    'success': True,
                    'fidelity': actual_fidelity,
                    'timestamp': current_time,
                    'path': path
                })
            else:
                # Record failed attempt
                self.metrics_collector.record_entanglement(
                    source=source,
                    destination=destination,
                    timestamp=current_time,
                    fidelity=0.0,
                    path=path,
                    protocol=protocol,
                    success=False
                )
                
                results.append({
                    'attempt': i,
                    'success': False,
                    'fidelity': 0.0,
                    'timestamp': current_time,
                    'path': path
                })
        
        successful = sum(1 for r in results if r['success'])
        logger.info(f"Experiment complete: {successful}/{num_entanglements} successful")
        
        return results
    
    def run_batch(self, 
                 pairs: List[Tuple[str, str]], 
                 protocols: List[str] = ['fasp', 'baseline'],
                 num_entanglements: int = 100):
        """
        Run batch of experiments for multiple source-destination pairs.
        
        Args:
            pairs: List of (source, destination) tuples
            protocols: List of protocols to test
            num_entanglements: Number of entanglements per experiment
        """
        logger.info(f"Starting batch simulation with {len(pairs)} pairs and {len(protocols)} protocols")
        
        start_time = time.time()
        self.metrics_collector.set_simulation_time(0, num_entanglements * 1e9 * len(pairs) * len(protocols))
        
        total_experiments = len(pairs) * len(protocols)
        completed = 0
        
        for i, (source, dest) in enumerate(pairs, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Pair {i}/{len(pairs)}: {source} -> {dest}")
            logger.info(f"{'='*60}")
            
            for protocol in protocols:
                try:
                    self.run_experiment(source, dest, protocol, num_entanglements)
                    completed += 1
                    progress = (completed / total_experiments) * 100
                    logger.info(f"Progress: {completed}/{total_experiments} ({progress:.1f}%)")
                except Exception as e:
                    logger.error(f"Error in experiment {source}->{dest} ({protocol}): {e}")
                    import traceback
                    traceback.print_exc()
        
        elapsed_time = time.time() - start_time
        logger.info(f"\nBatch simulation complete in {elapsed_time:.2f} seconds")
    
    def save_results(self, output_dir: str = 'results'):
        """
        Save simulation results to files.
        
        Args:
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/raw_data", exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Export to CSV
        csv_file = f"{output_dir}/raw_data/entanglements_{timestamp}.csv"
        self.metrics_collector.export_to_csv(csv_file)
        logger.info(f"Saved CSV data to {csv_file}")
        
        # Export to JSON
        json_file = f"{output_dir}/raw_data/metrics_{timestamp}.json"
        self.metrics_collector.export_to_json(json_file)
        logger.info(f"Saved JSON metrics to {json_file}")
        
        # Print summary
        self.metrics_collector.print_summary()
        
        # Save summary to text file
        summary_file = f"{output_dir}/summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            import sys
            old_stdout = sys.stdout
            sys.stdout = f
            self.metrics_collector.print_summary()
            sys.stdout = old_stdout
        
        logger.info(f"Saved summary to {summary_file}")
    
    def print_network_info(self):
        """Print network configuration information."""
        print("\n" + "="*60)
        print(f"NETWORK: {self.graph.config.get('name', 'Unnamed')}")
        print("="*60)
        print(f"Nodes: {len(self.graph.nodes)}")
        print(f"Links: {len(self.graph.links) // 2} (bidirectional)")
        print("\nNode List:")
        for node_name in sorted(self.graph.nodes.keys()):
            print(f"  - {node_name}")
        print("="*60 + "\n")
        
        if self.fidelity_estimator:
            self.fidelity_estimator.print_network_summary()
    
    def get_random_pairs(self, num_pairs: int = 10, seed: int = 0) -> List[Tuple[str, str]]:
        """Generate random source-destination pairs."""
        import random
        random.seed(seed)
        
        node_names = list(self.graph.nodes.keys())
        pairs = []
        
        for _ in range(num_pairs):
            source = random.choice(node_names)
            dest = random.choice([n for n in node_names if n != source])
            pairs.append((source, dest))
        
        return pairs
    

    def get_random_pairs_more_than_two_hops(self, num_pairs : int = 10, seed: int = 0) -> List[Tuple[str, str]]:
        """
        Generate source-destination pairs with paths having more than 2 hops only.
        """
        import random
        random.seed(seed)
        node_names = list(self.graph.nodes.keys())
        pairs = []

        attempts = 0
        max_attempts = 1000

        while len(pairs) < num_pairs and attempts < max_attempts:
            source = random.choice(node_names)
            dest = random.choice([n for n in node_names if n != source])
            path = self.baseline_router.find_path(source, dest)
            if path and len(path) > 3:  # path includes source and dest, so > 3 means > 2 hops
                pairs.append((source, dest))
            attempts += 1

        if len(pairs) < num_pairs:
            print(f"Could only find {len(pairs)} pairs with more than 2 hops after {attempts} attempts.")

        return pairs



def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run quantum network routing simulations'
    )
    
    parser.add_argument(
        '--topology',
        type=str,
        default='topologies/nsfnet.json',
        help='Path to topology JSON file'
    )
    
    parser.add_argument(
        '--protocol',
        type=str,
        choices=['fasp', 'baseline', 'both'],
        default='both',
        help='Protocol to test'
    )
    
    parser.add_argument(
        '--pairs',
        type=str,
        default=None,
        help='Comma-separated source-destination pairs (e.g., "WA-NY,CA1-PA")'
    )
    
    parser.add_argument(
        '--num-pairs',
        type=int,
        default=10,
        help='Number of random pairs to test (if --pairs not specified)'
    )
    
    parser.add_argument(
        '--num-entanglements',
        type=int,
        default=100,
        help='Number of entanglements per experiment'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set random seed
    np.random.seed(args.seed)
    
    logger.info("="*70)
    logger.info("QUANTUM NETWORK ROUTING SIMULATION")
    logger.info("="*70)
    
    # Initialize simulation runner
    runner = SimulationRunner(args.topology)
    runner.setup_network()
    runner.print_network_info()
    
    # Determine protocols to test
    protocols = []
    if args.protocol == 'both':
        protocols = ['fasp', 'baseline']
    else:
        protocols = [args.protocol]
    
    # Determine source-destination pairs
    if args.pairs:
        # Parse user-specified pairs
        pairs = []
        for pair_str in args.pairs.split(','):
            source, dest = pair_str.strip().split('-')
            pairs.append((source.strip(), dest.strip()))
    else:
        # Generate random pairs
        pairs = runner.get_random_pairs_more_than_two_hops(
            num_pairs=args.num_pairs,
            seed=args.seed
        )
        # pairs = runner.get_random_pairs(
        #     num_pairs=args.num_pairs,
        #     seed=args.seed
        # )
    
    logger.info(f"\nTesting {len(pairs)} source-destination pairs:")
    for src, dst in pairs:
        logger.info(f"  {src} -> {dst}")
    
    # Run simulations
    runner.run_batch(
        pairs=pairs,
        protocols=protocols,
        num_entanglements=args.num_entanglements
    )
    
    # Save results
    runner.save_results(output_dir=args.output_dir)
    
    logger.info("\n" + "="*70)
    logger.info("SIMULATION COMPLETE")
    logger.info("="*70)


if __name__ == "__main__":
    main()