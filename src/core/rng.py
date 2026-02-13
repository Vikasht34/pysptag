"""
RNG - Relative Neighborhood Graph
Direct Python port from SPTAG C++ implementation
Supports L2, InnerProduct, and Cosine metrics
"""
import numpy as np
from typing import List, Tuple, Literal

MetricType = Literal['L2', 'IP', 'Cosine']


class RNG:
    """
    Relative Neighborhood Graph
    Exact port from SPTAG/AnnService/inc/Core/Common/RelativeNeighborhoodGraph.h
    
    RNG condition: Keep edge (u,v) if for all w:
        d(u,v) <= RNGFactor * max(d(u,w), d(v,w))
    """
    
    def __init__(
        self,
        neighborhood_size: int = 32,
        rng_factor: float = 1.0,
        metric: MetricType = 'L2'
    ):
        self.neighborhood_size = neighborhood_size
        self.rng_factor = rng_factor
        self.metric = metric
        self.graph: List[np.ndarray] = []
    
    def _compute_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute distance based on metric"""
        if self.metric == 'L2':
            return np.sum((a - b) ** 2)
        elif self.metric == 'IP':
            return -np.dot(a, b)
        elif self.metric == 'Cosine':
            return -np.dot(a, b)
        return 0.0
        
    def build(self, data: np.ndarray, init_graph: List[np.ndarray] = None):
        """
        Build RNG from data
        
        Args:
            data: (N, D) array
            init_graph: Optional initial graph (from BKTree search)
        """
        n = len(data)
        
        if init_graph is None:
            # Initialize with empty graph
            self.graph = [np.full(self.neighborhood_size, -1, dtype=np.int32) for _ in range(n)]
        else:
            self.graph = [g.copy() for g in init_graph]
        
        # Refine with RNG condition
        for i in range(n):
            self._rebuild_neighbors(data, i, self.graph[i])
    
    def _rebuild_neighbors(
        self,
        data: np.ndarray,
        node: int,
        candidates: np.ndarray
    ):
        """
        Rebuild neighbors for a node using RNG condition
        
        Args:
            data: (N, D) dataset
            node: Node index
            candidates: Candidate neighbors (sorted by distance)
        """
        neighbors = []
        node_vec = data[node]
        
        for candidate in candidates:
            if candidate < 0 or candidate == node:
                continue
            
            if len(neighbors) >= self.neighborhood_size:
                break
            
            candidate_vec = data[candidate]
            candidate_dist = self._compute_distance(node_vec, candidate_vec)
            
            # Check RNG condition against existing neighbors
            is_good = True
            for neighbor in neighbors:
                neighbor_vec = data[neighbor]
                neighbor_dist = self._compute_distance(candidate_vec, neighbor_vec)
                
                # RNG condition: d(node, candidate) should be <= RNGFactor * d(neighbor, candidate)
                if self.rng_factor * neighbor_dist < candidate_dist:
                    is_good = False
                    break
            
            if is_good:
                neighbors.append(candidate)
        
        # Update graph
        self.graph[node][:len(neighbors)] = neighbors
        self.graph[node][len(neighbors):] = -1
    
    def insert_neighbor(
        self,
        data: np.ndarray,
        node: int,
        insert_node: int,
        insert_dist: float
    ):
        """
        Insert a neighbor into node's neighbor list
        
        Args:
            data: (N, D) dataset
            node: Node to update
            insert_node: Node to insert
            insert_dist: Distance from node to insert_node
        """
        neighbors = self.graph[node]
        node_vec = data[node]
        insert_vec = data[insert_node]
        
        # Find insertion position
        for k in range(self.neighborhood_size):
            current = neighbors[k]
            
            if current < 0:
                # Empty slot
                neighbors[k] = insert_node
                break
            
            current_vec = data[current]
            current_dist = self._compute_distance(node_vec, current_vec)
            
            if current_dist > insert_dist or (insert_dist == current_dist and insert_node < current):
                # Insert here
                neighbors[k] = insert_node
                
                # Shift remaining and check RNG condition
                k += 1
                while k < self.neighborhood_size:
                    if current < 0:
                        break
                    
                    current_vec = data[current]
                    current_to_node = self._compute_distance(current_vec, node_vec)
                    current_to_insert = self._compute_distance(current_vec, insert_vec)
                    
                    if current_to_node <= current_to_insert:
                        neighbors[k], current = current, neighbors[k]
                        k += 1
                    else:
                        break
                break
            
            # Check if insert_node violates RNG with current
            current_to_insert = self._compute_distance(current_vec, insert_vec)
            if current_to_insert < insert_dist:
                break
    
    def search(
        self,
        query: np.ndarray,
        data: np.ndarray,
        entry_point: int,
        k: int,
        max_check: int = -1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Greedy search on RNG
        
        Args:
            query: (D,) query vector
            data: (N, D) dataset
            entry_point: Starting node
            k: Number of neighbors
            max_check: Max nodes to check
            
        Returns:
            distances: (k,) distances
            indices: (k,) indices
        """
        if max_check < 0:
            max_check = len(data)
        
        visited = set()
        candidates = []
        
        # Start from entry point
        dist = self._compute_distance(query, data[entry_point])
        candidates.append((dist, entry_point))
        visited.add(entry_point)
        
        checked = 0
        
        while candidates and checked < max_check:
            # Get closest unvisited
            candidates.sort()
            dist, current = candidates.pop(0)
            checked += 1
            
            # Check neighbors
            for neighbor in self.graph[current]:
                if neighbor < 0 or neighbor in visited:
                    continue
                
                visited.add(neighbor)
                neighbor_dist = self._compute_distance(query, data[neighbor])
                candidates.append((neighbor_dist, neighbor))
        
        # Get top-k from visited
        results = []
        for node in visited:
            dist = self._compute_distance(query, data[node])
            results.append((dist, node))
        
        results.sort()
        results = results[:k]
        
        distances = np.array([d for d, _ in results])
        indices = np.array([idx for _, idx in results])
        
        return distances, indices
