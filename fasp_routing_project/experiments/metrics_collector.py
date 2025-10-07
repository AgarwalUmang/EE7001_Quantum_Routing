"""
Metrics Collector for Quantum Network Simulation
Task T2.4 - Member 3

Collects and analyzes performance metrics from simulation runs.
Tracks fidelity, throughput, and memory utilization.
"""

import json
import csv
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EntanglementRecord:
    """Record of a single entanglement establishment event."""
    source: str
    destination: str
    timestamp: float
    fidelity: float
    path: List[str]
    hops: int
    protocol: str  # 'fasp' or 'baseline'
    success: bool


@dataclass
class MemoryUtilization:
    """Memory usage statistics for a node."""
    node_name: str
    total_occupancy_time: float
    num_operations: int
    avg_occupancy_time: float
    max_occupancy_time: float
    utilization_percentage: float


@dataclass
class PathMetrics:
    """Aggregated metrics for a source-destination pair."""
    source: str
    destination: str
    protocol: str
    num_attempts: int
    num_successes: int
    success_rate: float
    avg_fidelity: float
    std_fidelity: float
    min_fidelity: float
    max_fidelity: float
    avg_hops: float
    avg_establishment_time: float
    throughput: float  # Entanglements per second


class MetricsCollector:
    """
    Collects and analyzes performance metrics from quantum network simulations.
    """
    
    def __init__(self):
        """Initialize the metrics collector."""
        self.entanglement_records = []
        self.memory_events = defaultdict(list)
        self.simulation_start_time = 0
        self.simulation_end_time = 0
        
        logger.info("MetricsCollector initialized")
    
    def record_entanglement(self,
                           source: str,
                           destination: str,
                           timestamp: float,
                           fidelity: float,
                           path: List[str],
                           protocol: str,
                           success: bool = True):
        """
        Record an entanglement establishment attempt.
        
        Args:
            source: Source node name
            destination: Destination node name
            timestamp: Simulation time of establishment
            fidelity: End-to-end fidelity achieved
            path: List of nodes in the path
            protocol: Protocol used ('fasp' or 'baseline')
            success: Whether entanglement was successfully established
        """
        record = EntanglementRecord(
            source=source,
            destination=destination,
            timestamp=timestamp,
            fidelity=fidelity,
            path=path,
            hops=len(path) - 1,
            protocol=protocol,
            success=success
        )
        
        self.entanglement_records.append(record)
        
        logger.debug(f"Recorded entanglement: {source}->{destination}, "
                    f"fidelity={fidelity:.4f}, protocol={protocol}")
    
    def record_memory_usage(self,
                           node_name: str,
                           start_time: float,
                           end_time: float,
                           operation_type: str = "storage"):
        """
        Record memory usage event.
        
        Args:
            node_name: Name of the node
            start_time: Time when qubit entered memory
            end_time: Time when qubit left memory
            operation_type: Type of operation (storage, swap, etc.)
        """
        occupancy_time = end_time - start_time
        
        self.memory_events[node_name].append({
            'start_time': start_time,
            'end_time': end_time,
            'occupancy_time': occupancy_time,
            'operation_type': operation_type
        })
        
        logger.debug(f"Recorded memory usage: {node_name}, "
                    f"occupancy={occupancy_time/1e9:.3f}ms")
    
    def set_simulation_time(self, start_time: float, end_time: float):
        """
        Set simulation time boundaries.
        
        Args:
            start_time: Simulation start time
            end_time: Simulation end time
        """
        self.simulation_start_time = start_time
        self.simulation_end_time = end_time
        
        duration = (end_time - start_time) / 1e12  # Convert to seconds
        logger.info(f"Simulation duration: {duration:.2f} seconds")
    
    def get_path_metrics(self, protocol: str = None) -> List[PathMetrics]:
        """
        Calculate aggregated metrics for each source-destination pair.
        
        Args:
            protocol: Filter by protocol ('fasp', 'baseline', or None for all)
            
        Returns:
            List of PathMetrics objects
        """
        # Group records by (source, destination, protocol)
        grouped = defaultdict(list)
        
        for record in self.entanglement_records:
            if protocol is None or record.protocol == protocol:
                key = (record.source, record.destination, record.protocol)
                grouped[key].append(record)
        
        # Calculate metrics for each group
        metrics_list = []
        
        for (source, dest, proto), records in grouped.items():
            successful = [r for r in records if r.success]
            
            if not records:
                continue
            
            # Calculate statistics
            fidelities = [r.fidelity for r in successful]
            hops = [r.hops for r in successful]
            times = [r.timestamp for r in successful]
            
            num_successes = len(successful)
            num_attempts = len(records)
            
            # Calculate throughput
            if times:
                time_span = (max(times) - min(times)) / 1e12  # Convert to seconds
                throughput = num_successes / max(time_span, 1e-6)
            else:
                throughput = 0.0
            
            # Calculate establishment time
            if len(times) > 1:
                time_diffs = np.diff(sorted(times))
                avg_est_time = np.mean(time_diffs) / 1e9 if len(time_diffs) > 0 else 0
            else:
                avg_est_time = 0
            
            metrics = PathMetrics(
                source=source,
                destination=dest,
                protocol=proto,
                num_attempts=num_attempts,
                num_successes=num_successes,
                success_rate=num_successes / num_attempts if num_attempts > 0 else 0,
                avg_fidelity=np.mean(fidelities) if fidelities else 0,
                std_fidelity=np.std(fidelities) if fidelities else 0,
                min_fidelity=min(fidelities) if fidelities else 0,
                max_fidelity=max(fidelities) if fidelities else 0,
                avg_hops=np.mean(hops) if hops else 0,
                avg_establishment_time=avg_est_time,
                throughput=throughput
            )
            
            metrics_list.append(metrics)
        
        return metrics_list
    
    def get_memory_utilization(self) -> List[MemoryUtilization]:
        """
        Calculate memory utilization statistics for each node.
        
        Returns:
            List of MemoryUtilization objects
        """
        utilization_list = []
        
        sim_duration = self.simulation_end_time - self.simulation_start_time
        
        for node_name, events in self.memory_events.items():
            if not events:
                continue
            
            occupancy_times = [e['occupancy_time'] for e in events]
            total_occupancy = sum(occupancy_times)
            
            # Calculate utilization percentage
            utilization_pct = (total_occupancy / sim_duration) * 100 if sim_duration > 0 else 0
            
            util = MemoryUtilization(
                node_name=node_name,
                total_occupancy_time=total_occupancy,
                num_operations=len(events),
                avg_occupancy_time=np.mean(occupancy_times),
                max_occupancy_time=max(occupancy_times),
                utilization_percentage=utilization_pct
            )
            
            utilization_list.append(util)
        
        return utilization_list
    
    def compare_protocols(self) -> Dict:
        """
        Compare FASP and baseline protocols across all metrics.
        
        Returns:
            Dictionary with comparison statistics
        """
        fasp_metrics = self.get_path_metrics(protocol='fasp')
        baseline_metrics = self.get_path_metrics(protocol='baseline')
        
        # Calculate averages
        def avg_metric(metrics_list, attr):
            values = [getattr(m, attr) for m in metrics_list]
            return np.mean(values) if values else 0
        
        comparison = {
            'fasp': {
                'avg_fidelity': avg_metric(fasp_metrics, 'avg_fidelity'),
                'avg_hops': avg_metric(fasp_metrics, 'avg_hops'),
                'avg_throughput': avg_metric(fasp_metrics, 'throughput'),
                'avg_success_rate': avg_metric(fasp_metrics, 'success_rate'),
                'num_paths': len(fasp_metrics)
            },
            'baseline': {
                'avg_fidelity': avg_metric(baseline_metrics, 'avg_fidelity'),
                'avg_hops': avg_metric(baseline_metrics, 'avg_hops'),
                'avg_throughput': avg_metric(baseline_metrics, 'throughput'),
                'avg_success_rate': avg_metric(baseline_metrics, 'success_rate'),
                'num_paths': len(baseline_metrics)
            }
        }
        
        # Calculate improvements
        fasp_fid = comparison['fasp']['avg_fidelity']
        base_fid = comparison['baseline']['avg_fidelity']
        
        if base_fid > 0:
            comparison['improvement'] = {
                'fidelity_gain': fasp_fid - base_fid,
                'fidelity_gain_pct': ((fasp_fid / base_fid) - 1) * 100,
                'hop_difference': comparison['fasp']['avg_hops'] - comparison['baseline']['avg_hops'],
                'throughput_ratio': comparison['fasp']['avg_throughput'] / max(comparison['baseline']['avg_throughput'], 1e-6)
            }
        
        return comparison
    
    def export_to_csv(self, filename: str):
        """
        Export entanglement records to CSV file.
        
        Args:
            filename: Output CSV filename
        """
        if not self.entanglement_records:
            logger.warning("No entanglement records to export")
            return
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['source', 'destination', 'timestamp', 'fidelity', 
                         'hops', 'protocol', 'success', 'path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for record in self.entanglement_records:
                row = asdict(record)
                row['path'] = '->'.join(record.path)
                writer.writerow(row)
        
        logger.info(f"Exported {len(self.entanglement_records)} records to {filename}")
    
    def export_to_json(self, filename: str):
        """
        Export all metrics to JSON file.
        
        Args:
            filename: Output JSON filename
        """
        data = {
            'entanglement_records': [asdict(r) for r in self.entanglement_records],
            'path_metrics': {
                'fasp': [asdict(m) for m in self.get_path_metrics('fasp')],
                'baseline': [asdict(m) for m in self.get_path_metrics('baseline')]
            },
            'memory_utilization': [asdict(m) for m in self.get_memory_utilization()],
            'comparison': self.compare_protocols(),
            'simulation_info': {
                'start_time': self.simulation_start_time,
                'end_time': self.simulation_end_time,
                'duration': (self.simulation_end_time - self.simulation_start_time) / 1e12
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported metrics to {filename}")
    
    def print_summary(self):
        """Print a summary of collected metrics."""
        print("\n" + "="*70)
        print("SIMULATION METRICS SUMMARY")
        print("="*70)
        
        # Protocol comparison
        comparison = self.compare_protocols()
        
        print("\nProtocol Comparison:")
        print("-" * 70)
        print(f"{'Metric':<30} {'FASP':<20} {'Baseline':<20}")
        print("-" * 70)
        print(f"{'Average Fidelity':<30} {comparison['fasp']['avg_fidelity']:.4f}           "
              f"{comparison['baseline']['avg_fidelity']:.4f}")
        print(f"{'Average Hops':<30} {comparison['fasp']['avg_hops']:.2f}             "
              f"{comparison['baseline']['avg_hops']:.2f}")
        print(f"{'Average Throughput (ent/s)':<30} {comparison['fasp']['avg_throughput']:.4f}         "
              f"{comparison['baseline']['avg_throughput']:.4f}")
        print(f"{'Success Rate':<30} {comparison['fasp']['avg_success_rate']:.2%}          "
              f"{comparison['baseline']['avg_success_rate']:.2%}")
        
        if 'improvement' in comparison:
            print("\nFASP Improvements:")
            print("-" * 70)
            print(f"Fidelity Gain: {comparison['improvement']['fidelity_gain']:+.4f} "
                  f"({comparison['improvement']['fidelity_gain_pct']:+.2f}%)")
            print(f"Hop Difference: {comparison['improvement']['hop_difference']:+.2f}")
            print(f"Throughput Ratio: {comparison['improvement']['throughput_ratio']:.2f}x")
        
        # Memory utilization
        mem_utils = self.get_memory_utilization()
        if mem_utils:
            print("\nMemory Utilization (Top 5 Nodes):")
            print("-" * 70)
            sorted_utils = sorted(mem_utils, key=lambda x: x.utilization_percentage, reverse=True)[:5]
            for util in sorted_utils:
                print(f"{util.node_name:<10} Utilization: {util.utilization_percentage:>6.2f}% "
                      f"Avg Occupancy: {util.avg_occupancy_time/1e9:>8.2f}ms")
        
        print("="*70 + "\n")
    
    def clear(self):
        """Clear all collected data."""
        self.entanglement_records.clear()
        self.memory_events.clear()
        self.simulation_start_time = 0
        self.simulation_end_time = 0
        logger.info("Metrics collector cleared")


# Test function
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("MetricsCollector module loaded")
    print("\nUsage example:")
    print("  collector = MetricsCollector()")
    print("  collector.record_entanglement('A', 'C', 1000, 0.95, ['A','B','C'], 'fasp')")
    print("  collector.print_summary()")