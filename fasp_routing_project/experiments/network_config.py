"""
Network Configuration Utility
Handles loading topology files and creating SeQUeNCe network instances.
"""

import json
import logging
from typing import Dict, List, Tuple
import sys
import os

# Add SeQUeNCe to path
sys.path.insert(0, os.path.abspath('../'))

from sequence.kernel.timeline import Timeline
from sequence.topology.node import QuantumRouter
from sequence.components.memory import Memory
from sequence.components.optical_channel import QuantumChannel, ClassicalChannel
from sequence.components.bsm import SingleAtomBSM
from sequence.entanglement_management.generation import EntanglementGenerationA
from sequence.network_management.reservation import ResourceReservationProtocol

logger = logging.getLogger(__name__)


class NetworkConfiguration:
    """
    Configures and builds SeQUeNCe quantum networks from JSON topology files.
    """
    
    def __init__(self, topology_file: str):
        """
        Initialize network configuration.
        
        Args:
            topology_file: Path to JSON topology configuration file
        """
        self.topology_file = topology_file
        self.config = None
        self.timeline = None
        self.network = None
        
        self._load_config()
        logger.info(f"Loaded topology: {self.config.get('name', 'Unnamed')}")
    
    def _load_config(self):
        """Load topology configuration from JSON file."""
        try:
            with open(self.topology_file, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            logger.error(f"Topology file not found: {self.topology_file}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in topology file: {e}")
            raise
    
    def build_network(self) -> Timeline:
        """
        Build complete quantum network from configuration.
        
        Returns:
            SeQUeNCe Timeline object with configured network
        """
        # Create timeline
        sim_config = self.config.get('simulation_config', {})
        stop_time = sim_config.get('stop_time', 10e12)
        seed = sim_config.get('seed', 0)
        
        self.timeline = Timeline(stop_time=stop_time, formalism='density_matrix')
        self.timeline.seed(seed)
        
        logger.info(f"Created timeline with stop_time={stop_time/1e12:.1f}s")
        
        # Create nodes
        self._create_nodes()
        
        # Create quantum connections
        self._create_quantum_channels()
        
        # Create classical connections
        self._create_classical_channels()
        
        # Initialize protocols
        self._initialize_protocols()
        
        # Initialize timeline
        self.timeline.init()
        
        logger.info("Network built successfully")
        
        return self.timeline
    
    def _create_nodes(self):
        """Create quantum router nodes."""
        nodes_config = self.config.get('nodes', [])
        hardware_config = self.config.get('hardware_config', {})
        
        for node_config in nodes_config:
            node_name = node_config['name']
            node_type = node_config.get('type', 'QuantumRouter')
            seed = node_config.get('seed', 0)
            
            if node_type == 'QuantumRouter':
                node = QuantumRouter(node_name, self.timeline)
                # self.timeline.add_entity(node)  # <-- REMOVED: already added in QuantumRouter constructor
                # Configure memory properties
                memory_config = hardware_config.get('memory', {})
                num_memories = node_config.get('memo_size', 50)
                for i in range(num_memories):
                    mem = Memory(
                        f"{node_name}_mem_{i}",
                        self.timeline,
                        fidelity=memory_config.get('fidelity', 0.995),
                        frequency=memory_config.get('frequency', 2000),
                        efficiency=memory_config.get('efficiency', 1.0),
                        coherence_time=memory_config.get('coherence_time', 1e12),
                        wavelength=memory_config.get('wavelength', 1550e-9)
                    )
                    node.add_component(mem)
                logger.debug(f"Created node: {node_name} with {num_memories} memories")
    
    def _create_quantum_channels(self):
        """Create quantum channels between nodes."""
        qc_config_list = self.config.get('quantum_connections', [])
        hardware_config = self.config.get('hardware_config', {})
        qc_hardware = hardware_config.get('quantum_channel', {})

        created_channels = set()  # Track created channel names

        for qc_config in qc_config_list:
            node_names = qc_config['nodes']
            distance = qc_config.get('distance', 1e3)  # Default 1 km
            attenuation = qc_config.get('attenuation', qc_hardware.get('attenuation', 0.0002))

            node1 = self.timeline.get_entity_by_name(node_names[0])
            node2 = self.timeline.get_entity_by_name(node_names[1])

            # Only create channel in the direction specified
            qc_name = f"qc_{node_names[0]}_to_{node_names[1]}"
            if qc_name not in created_channels:
                QuantumChannel(
                    qc_name,
                    self.timeline,
                    attenuation=attenuation,
                    distance=distance,
                    polarization_fidelity=qc_hardware.get('polarization_fidelity', 0.99)
                ).set_ends(node1, node2.name)
                created_channels.add(qc_name)
                logger.debug(f"Created quantum channel: {qc_name}")

    def _create_classical_channels(self):
        """Create classical communication channels."""
        cc_config_list = self.config.get('classical_connections', [])
        
        for cc_config in cc_config_list:
            node_names = cc_config['nodes']
            distance = cc_config.get('distance', 1e3)
            delay = cc_config.get('delay', distance / 200000)  # Speed of light in fiber
            
            node1 = self.timeline.get_entity_by_name(node_names[0])
            node2 = self.timeline.get_entity_by_name(node_names[1])
            
            # Create bidirectional classical channels
            cc1_name = f"cc_{node_names[0]}_to_{node_names[1]}"
            cc2_name = f"cc_{node_names[1]}_to_{node_names[0]}"
            
            cc1 = ClassicalChannel(cc1_name, self.timeline, distance=distance, delay=delay)
            cc1.set_ends(node1, node2.name)
            
            cc2 = ClassicalChannel(cc2_name, self.timeline, distance=distance, delay=delay)
            cc2.set_ends(node2, node1.name)
            
            logger.debug(f"Created classical channels: {node_names[0]} <-> {node_names[1]}")
    
    def _initialize_protocols(self):
        """Initialize entanglement management protocols on all nodes."""
        protocol_config = self.config.get('protocol_stack', {})
        
        for entity in self.timeline.entities:
            if isinstance(entity, QuantumRouter):
                # Initialize resource management
                entity.network_manager.protocol_stack = []
                
                # Add entanglement generation protocols for each quantum channel
                for qc_name in entity.qchannels:
                    qc = entity.qchannels[qc_name]
                    if hasattr(qc, 'receiver'):
                        # Create entanglement generation protocol
                        eg_protocol = EntanglementGenerationA(
                            entity,
                            f"EG_{qc_name}",
                            middle=qc.receiver.owner.name,
                            other=qc.receiver.owner.name,
                            memory=entity.memory_array[0]
                        )
                        
                        entity.protocols.append(eg_protocol)
                
                logger.debug(f"Initialized protocols on node: {entity.name}")
    
    def get_node_names(self) -> List[str]:
        """Get list of all node names in network."""
        return [node['name'] for node in self.config.get('nodes', [])]
    
    def get_random_pairs(self, num_pairs: int = 10, seed: int = 0) -> List[Tuple[str, str]]:
        """
        Generate random source-destination pairs for testing.
        
        Args:
            num_pairs: Number of pairs to generate
            seed: Random seed
            
        Returns:
            List of (source, dest) tuples
        """
        import random
        random.seed(seed)
        
        node_names = self.get_node_names()
        pairs = []
        
        for _ in range(num_pairs):
            source = random.choice(node_names)
            dest = random.choice([n for n in node_names if n != source])
            pairs.append((source, dest))
        
        return pairs
    
    def get_all_pairs(self) -> List[Tuple[str, str]]:
        """Get all possible source-destination pairs."""
        node_names = self.get_node_names()
        pairs = []
        
        for i, source in enumerate(node_names):
            for dest in node_names[i+1:]:
                pairs.append((source, dest))
        
        return pairs
    
    def print_network_summary(self):
        """Print summary of network configuration."""
        print("\n" + "="*60)
        print(f"NETWORK: {self.config.get('name', 'Unnamed')}")
        print("="*60)
        print(f"Nodes: {len(self.config.get('nodes', []))}")
        print(f"Quantum Connections: {len(self.config.get('quantum_connections', []))}")
        print(f"Classical Connections: {len(self.config.get('classical_connections', []))}")
        print("\nNode List:")
        for node in self.config.get('nodes', []):
            print(f"  - {node['name']}")
        print("="*60 + "\n")


def create_simple_network(num_nodes: int = 3, distance: float = 50e3) -> Timeline:
    """
    Create a simple linear network for testing.
    """
    timeline = Timeline(stop_time=10e12, formalism='density_matrix')
    timeline.seed(0)

    # Create nodes
    nodes = []
    for i in range(num_nodes):
        node = QuantumRouter(f"Node{i}", timeline, memo_size=10)
        nodes.append(node)
        # timeline.entities.append(node)  # <-- REMOVE THIS LINE

    # Create connections
    for i in range(num_nodes - 1):
        # Quantum channels
        qc1 = QuantumChannel(f"qc{i}to{i+1}", timeline, attenuation=0.0002, distance=distance)
        qc1.set_ends(nodes[i], nodes[i+1].name)

        qc2 = QuantumChannel(f"qc{i+1}to{i}", timeline, attenuation=0.0002, distance=distance)
        qc2.set_ends(nodes[i+1], nodes[i].name)

        # Classical channels
        cc1 = ClassicalChannel(f"cc{i}to{i+1}", timeline, distance=distance)
        cc1.set_ends(nodes[i], nodes[i+1].name)

        cc2 = ClassicalChannel(f"cc{i+1}to{i}", timeline, distance=distance)
        cc2.set_ends(nodes[i+1], nodes[i].name)

    timeline.init()

    logger.info(f"Created simple {num_nodes}-node linear network")

    return timeline


# Test function
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("NetworkConfiguration module loaded")
    print("Usage example:")
    print("  config = NetworkConfiguration('topologies/nsfnet.json')")
    print("  timeline = config.build_network()")
    print("  config.print_network_summary()")