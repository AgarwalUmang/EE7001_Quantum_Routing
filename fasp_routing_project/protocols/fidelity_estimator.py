"""
FIXED Fidelity Estimator Class for Quantum Network Routing
Now properly calculates distance-dependent fidelity

This module calculates expected link fidelities based on physical hardware parameters.
"""

import math
import numpy as np
from typing import Dict, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FidelityEstimator:
    """
    Estimates the fidelity of entanglement links in a quantum network.
    
    This class queries hardware parameters and calculates the expected fidelity
    of an entangled pair after generation, storage, and usage in swapping operations.
    """
    
    def __init__(self, network):
        """
        Initialize the FidelityEstimator.
        
        Args:
            network: Network object containing topology and hardware info
        """
        self.network = network
        self.link_fidelities = {}
        self.hardware_params = {}
        
        # Cache hardware parameters for all nodes
        self._cache_hardware_parameters()
        
        logger.info("FidelityEstimator initialized successfully")
    
    def _cache_hardware_parameters(self):
        """
        Cache hardware parameters from all network nodes.
        This improves performance by avoiding repeated queries.
        """
        for node_name, node in self.network.nodes.items():
            self.hardware_params[node_name] = {
                'memory_coherence_time': self._get_memory_coherence_time(node),
                'memory_fidelity': self._get_memory_fidelity(node),
                'gate_fidelity': self._get_gate_fidelity(node)
            }
        
        logger.info(f"Cached hardware parameters for {len(self.hardware_params)} nodes")
    
    def _get_memory_coherence_time(self, node) -> float:
        """
        Extract memory coherence time (T2) from node hardware.
        
        Args:
            node: Network node object
            
        Returns:
            Coherence time in seconds (default: 1.0 second)
        """
        try:
            if hasattr(node, 'memory_array') and node.memory_array:
                memory = node.memory_array[0]
                if hasattr(memory, 'coherence_time'):
                    # Convert ps to seconds
                    return memory.coherence_time / 1e12
        except (AttributeError, IndexError):
            pass
        
        return 1.0  # Default: 1 second
    
    def _get_memory_fidelity(self, node) -> float:
        """
        Extract memory read/write fidelity from node hardware.
        
        Args:
            node: Network node object
            
        Returns:
            Memory operation fidelity (default: 0.995)
        """
        try:
            if hasattr(node, 'memory_array') and node.memory_array:
                memory = node.memory_array[0]
                if hasattr(memory, 'fidelity'):
                    return memory.fidelity
        except (AttributeError, IndexError):
            pass
        
        return 0.995  # Default memory fidelity
    
    def _get_gate_fidelity(self, node) -> float:
        """
        Extract Bell State Measurement (BSM) gate fidelity from node.
        
        Args:
            node: Network node object
            
        Returns:
            Gate fidelity for entanglement swapping (default: 0.98)
        """
        try:
            if hasattr(node, 'bsm_node') and node.bsm_node:
                if hasattr(node.bsm_node, 'fidelity'):
                    return node.bsm_node.fidelity
        except AttributeError:
            pass
        
        return 0.98  # Default gate fidelity
    
    def calculate_link_fidelity(self, 
                            source: str, 
                            dest: str, 
                            distance: Optional[float] = None) -> float:
        """
            Calculate expected fidelity for a direct quantum link with
            nonlinear, noisy, and hardware-dependent degradation.
        """

        link_id = f"{source}-{dest}"
        if link_id in self.link_fidelities:
            return self.link_fidelities[link_id]

        if distance is None:
            distance = self._get_link_distance(source, dest)

        # --- Physical base parameters ---
        INITIAL_FIDELITY = 0.99
        PHOTON_LOSS_RATE_DB = 0.02  # dB/km
        AVERAGE_WAIT_TIME = 0.05  # seconds

        src_params = self.hardware_params.get(source, {})
        dst_params = self.hardware_params.get(dest, {})

        # --- Distance-dependent transmission loss ---
        loss_db = PHOTON_LOSS_RATE_DB * distance
        transmission_efficiency = 10 ** (-loss_db / 10)

        # --- Memory decoherence with nonlinear decay ---
        dest_coh = dst_params.get("memory_coherence_time", 1.0)
        nonlinear_decay = math.exp(-(AVERAGE_WAIT_TIME / dest_coh) ** 1.2)

        # --- Hardware imperfections ---
        mem_fid_src = src_params.get("memory_fidelity", 0.995)
        mem_fid_dst = dst_params.get("memory_fidelity", 0.995)
        gate_fid = 0.98 * (0.98 + 0.02 * np.random.rand())

        # --- Environmental noise factor (simulated) ---
        env_factor = 1.0 - 0.005 * np.sin(distance / 40.0) - 0.01 * np.random.rand()

        # --- Combine components ---
        fidelity = INITIAL_FIDELITY
        # Photon loss impact (stronger curvature for long links)
        fidelity *= (0.7 + 0.3 * transmission_efficiency ** 0.4)
        # Memory and gate interactions
        fidelity *= (mem_fid_src * mem_fid_dst * gate_fid)
        # Nonlinear decoherence
        fidelity *= nonlinear_decay
        # Environmental fluctuations
        fidelity *= env_factor

        # Clamp to valid range
        fidelity = max(0.25, min(0.99, fidelity))

        self.link_fidelities[link_id] = fidelity

        logger.debug(f"Link {link_id}: dist={distance:.1f} km, "
                    f"loss={loss_db:.2f} dB, fid={fidelity:.4f}")

        return fidelity

    
    def _get_link_distance(self, source: str, dest: str) -> float:
        """
        Retrieve physical distance between two nodes from topology.
        
        Args:
            source: Source node name
            dest: Destination node name
            
        Returns:
            Distance in kilometers
        """
        try:
            # Try to get distance from quantum channel
            source_node = self.network.get_node(source)
            
            if hasattr(source_node, 'qchannels'):
                for qc_name, qc in source_node.qchannels.items():
                    if hasattr(qc, 'receiver') and hasattr(qc.receiver, 'owner'):
                        if qc.receiver.owner.name == dest:
                            # Distance is stored in meters, convert to km
                            if hasattr(qc, 'distance'):
                                distance_km = qc.distance / 1000
                                logger.debug(f"Found distance for {source}-{dest}: {distance_km:.1f} km")
                                return distance_km
            
            # Fallback: check if network has graph with link info
            if hasattr(self.network, 'graph') and hasattr(self.network.graph, 'links'):
                link_id = f"{source}-{dest}"
                if link_id in self.network.graph.links:
                    distance_km = self.network.graph.links[link_id]['distance']
                    logger.debug(f"Found distance from graph for {source}-{dest}: {distance_km:.1f} km")
                    return distance_km
        
        except Exception as e:
            logger.warning(f"Could not retrieve distance for {source}-{dest}: {e}")
        
        # Default fallback
        logger.warning(f"Using default distance for {source}-{dest}")
        return 100.0  # Default distance in km
    
    def calculate_path_fidelity(self, path: list) -> float:
        """
        Calculate expected end-to-end fidelity for a multi-hop path.
        
        For a path with entanglement swapping, the overall fidelity is approximately
        the product of individual link fidelities: F_path = ∏ F_link
        
        Args:
            path: List of node names forming the path
            
        Returns:
            Expected path fidelity
        """
        if len(path) < 2:
            logger.warning("Path too short for fidelity calculation")
            return 1.0
        
        path_fidelity = 1.0
        
        # Calculate fidelity for each link in path
        for i in range(len(path) - 1):
            link_fidelity = self.calculate_link_fidelity(path[i], path[i+1])
            path_fidelity *= link_fidelity
        
        logger.debug(f"Path {' -> '.join(path)}: fidelity={path_fidelity:.4f}")
        
        return path_fidelity
    
    def get_link_cost(self, source: str, dest: str) -> float:
        """
        Calculate the routing cost for a link based on fidelity.
        
        Uses logarithmic cost function: C(l) = -log(F_l)
        This transforms multiplicative fidelity into additive cost,
        allowing standard shortest-path algorithms to maximize fidelity.
        
        Args:
            source: Source node name
            dest: Destination node name
            
        Returns:
            Link cost (higher cost = lower fidelity)
        """
        fidelity = self.calculate_link_fidelity(source, dest)
        
        # Avoid log(0) by ensuring minimum fidelity
        fidelity = max(fidelity, 1e-6)
        
        cost = -math.log(fidelity)
        
        return cost
    
    def get_all_link_costs(self) -> Dict[Tuple[str, str], float]:
        """
        Pre-calculate costs for all links in the network.
        
        Returns:
            Dictionary mapping (source, dest) tuples to costs
        """
        link_costs = {}
        
        # Iterate over all nodes
        for node_name, node in self.network.nodes.items():
            # Find all quantum channels (links)
            if hasattr(node, 'qchannels'):
                for qc_name, qc in node.qchannels.items():
                    if hasattr(qc, 'receiver') and hasattr(qc.receiver, 'owner'):
                        dest_name = qc.receiver.owner.name
                        link_costs[(node_name, dest_name)] = self.get_link_cost(
                            node_name, dest_name
                        )
        
        logger.info(f"Calculated costs for {len(link_costs)} links")
        
        return link_costs
    
    def print_network_summary(self):
        """Print a summary of network fidelities for debugging."""
        print("\n" + "="*60)
        print("NETWORK FIDELITY SUMMARY")
        print("="*60)
        
        # Sort by fidelity to show variation
        sorted_links = sorted(self.link_fidelities.items(), 
                            key=lambda x: x[1], reverse=True)
        
        for link_id, fidelity in sorted_links:
            cost = -math.log(fidelity)
            # Get distance for display
            source, dest = link_id.split('-')
            distance = self._get_link_distance(source, dest)
            print(f"Link {link_id:15s}: Fidelity={fidelity:.4f}, "
                  f"Cost={cost:.4f}, Distance={distance:>6.1f}km")
        
        print("="*60 + "\n")
        
        # Print statistics
        fidelities = list(self.link_fidelities.values())
        if fidelities:
            print(f"Fidelity Statistics:")
            print(f"  Min:  {min(fidelities):.4f}")
            print(f"  Max:  {max(fidelities):.4f}")
            print(f"  Mean: {np.mean(fidelities):.4f}")
            print(f"  Std:  {np.std(fidelities):.4f}")
            print()


# Test function
if __name__ == "__main__":
    print("FidelityEstimator module loaded successfully")
    print("This module should be imported and used with a Network object")