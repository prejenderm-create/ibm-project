"""
Navigation & Traffic Routing System
------------------------------------
A simplified simulation of core Google-Maps-style functionality:
  1. Model a road network as a weighted graph (intersections + roads).
  2. Simulate live traffic congestion on each road.
  3. Predict the best (fastest) route between two points, factoring in traffic.
  4. Send congestion notifications when heavy traffic lies ahead on the chosen route.
  5. Suggest an alternate route if the current one becomes congested.

Dependencies: only Python's standard library (heapq, random, dataclasses, datetime).
Optional: networkx + matplotlib for visualizing the graph (see visualize_graph()).

Author: (your name here) — IBM mini-project
"""

import heapq
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------------------
# 1. ROAD NETWORK MODEL
# ---------------------------------------------------------------------------

@dataclass
class Road:
    """A directed road segment between two intersections."""
    to_node: str
    distance_km: float          # physical length of the road
    base_speed_kmh: float       # speed limit / free-flow speed
    congestion: float = 0.0     # 0.0 (clear) -> 1.0 (gridlock)

    def travel_time_minutes(self) -> float:
        """
        Effective travel time given current congestion.
        Congestion slows the effective speed down; at congestion = 1.0
        speed drops to 20% of free-flow speed (never fully zero, like real traffic).
        """
        effective_speed = self.base_speed_kmh * (1 - 0.8 * self.congestion)
        effective_speed = max(effective_speed, 1.0)  # avoid divide-by-zero
        return (self.distance_km / effective_speed) * 60


class RoadNetwork:
    """Graph of intersections (nodes) connected by roads (directed edges)."""

    def __init__(self):
        self.graph: Dict[str, List[Road]] = {}

    def add_intersection(self, name: str):
        self.graph.setdefault(name, [])

    def add_road(self, from_node: str, to_node: str, distance_km: float,
                 base_speed_kmh: float, bidirectional: bool = True):
        self.add_intersection(from_node)
        self.add_intersection(to_node)
        self.graph[from_node].append(Road(to_node, distance_km, base_speed_kmh))
        if bidirectional:
            self.graph[to_node].append(Road(from_node, distance_km, base_speed_kmh))

    def get_road(self, from_node: str, to_node: str) -> Optional[Road]:
        for road in self.graph.get(from_node, []):
            if road.to_node == to_node:
                return road
        return None


# ---------------------------------------------------------------------------
# 2. TRAFFIC SIMULATION (stands in for a live traffic-data feed / API)
# ---------------------------------------------------------------------------

class TrafficSimulator:
    """
    Randomly evolves congestion levels on every road each 'tick', simulating
    real-time traffic updates (rush hour, accidents, etc.). In a production
    system this would be replaced by a live feed (GPS probe data, sensors,
    or a third-party traffic API).
    """

    def __init__(self, network: RoadNetwork, volatility: float = 0.15):
        self.network = network
        self.volatility = volatility

    def tick(self):
        for roads in self.network.graph.values():
            for road in roads:
                drift = random.uniform(-self.volatility, self.volatility)
                road.congestion = min(1.0, max(0.0, road.congestion + drift))

    def simulate_incident(self, from_node: str, to_node: str, severity: float = 0.9):
        """Force heavy congestion on a specific road, e.g. an accident."""
        road = self.network.get_road(from_node, to_node)
        if road:
            road.congestion = severity


# ---------------------------------------------------------------------------
# 3. ROUTE PREDICTION (Dijkstra weighted by live travel time)
# ---------------------------------------------------------------------------

@dataclass
class Route:
    path: List[str]
    total_time_min: float
    total_distance_km: float
    segments: List[Road] = field(default_factory=list)


class RoutePredictor:
    def __init__(self, network: RoadNetwork):
        self.network = network

    def find_best_route(self, start: str, end: str) -> Optional[Route]:
        """Dijkstra's algorithm, weighted by current traffic-adjusted travel time."""
        if start not in self.network.graph or end not in self.network.graph:
            return None

        times = {node: float("inf") for node in self.network.graph}
        times[start] = 0
        prev: Dict[str, Tuple[str, Road]] = {}
        pq = [(0, start)]
        visited = set()

        while pq:
            current_time, node = heapq.heappop(pq)
            if node in visited:
                continue
            visited.add(node)
            if node == end:
                break

            for road in self.network.graph[node]:
                new_time = current_time + road.travel_time_minutes()
                if new_time < times[road.to_node]:
                    times[road.to_node] = new_time
                    prev[road.to_node] = (node, road)
                    heapq.heappush(pq, (new_time, road.to_node))

        if times[end] == float("inf"):
            return None

        # Reconstruct path
        path = [end]
        segments = []
        node = end
        while node != start:
            prev_node, road = prev[node]
            segments.append(road)
            path.append(prev_node)
            node = prev_node
        path.reverse()
        segments.reverse()

        total_distance = sum(r.distance_km for r in segments)
        return Route(path=path, total_time_min=times[end],
                      total_distance_km=total_distance, segments=segments)

    def find_alternate_route(self, start: str, end: str, avoid_road: Tuple[str, str]) -> Optional[Route]:
        """Recompute best route while temporarily blocking one congested road."""
        blocked_from, blocked_to = avoid_road
        original = self.network.get_road(blocked_from, blocked_to)
        if not original:
            return self.find_best_route(start, end)

        saved_congestion = original.congestion
        original.congestion = 1.0  # treat as effectively impassable
        try:
            alt = self.find_best_route(start, end)
        finally:
            original.congestion = saved_congestion
        return alt


# ---------------------------------------------------------------------------
# 4. CONGESTION NOTIFICATIONS
# ---------------------------------------------------------------------------

class NotificationService:
    CONGESTION_THRESHOLD = 0.6  # roads above this level trigger an alert

    @staticmethod
    def check_route(route: Route) -> List[str]:
        alerts = []
        for i, road in enumerate(route.segments):
            if road.congestion >= NotificationService.CONGESTION_THRESHOLD:
                from_node = route.path[i]
                to_node = route.path[i + 1]
                delay = road.travel_time_minutes() - (
                    road.distance_km / road.base_speed_kmh * 60
                )
                alerts.append(
                    f"⚠ Heavy traffic between {from_node} and {to_node} "
                    f"(congestion {road.congestion:.0%}) — est. delay +{delay:.1f} min"
                )
        return alerts


# ---------------------------------------------------------------------------
# 5. NAVIGATION CONTROLLER (ties everything together, like the Maps app itself)
# ---------------------------------------------------------------------------

class Navigator:
    def __init__(self, network: RoadNetwork):
        self.network = network
        self.simulator = TrafficSimulator(network)
        self.predictor = RoutePredictor(network)

    def get_directions(self, start: str, end: str) -> None:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Routing from '{start}' to '{end}'...")

        route = self.predictor.find_best_route(start, end)
        if not route:
            print("No route found.")
            return

        self._print_route("Best route", route)

        alerts = NotificationService.check_route(route)
        if alerts:
            print("\nTraffic notifications:")
            for a in alerts:
                print(" ", a)

            # If the very next segment is congested, offer an alternate route
            worst_road = max(route.segments, key=lambda r: r.congestion)
            idx = route.segments.index(worst_road)
            alt = self.predictor.find_alternate_route(
                start, end, (route.path[idx], route.path[idx + 1])
            )
            if alt and alt.total_time_min < route.total_time_min + 0.5:
                self._print_route("Suggested alternate route (avoids congestion)", alt)
        else:
            print("Traffic is clear — no delays expected on this route.")

    @staticmethod
    def _print_route(label: str, route: Route):
        print(f"\n{label}: {' -> '.join(route.path)}")
        print(f"  Distance: {route.total_distance_km:.1f} km")
        print(f"  ETA: {route.total_time_min:.1f} min")


# ---------------------------------------------------------------------------
# OPTIONAL: visualize the network with networkx + matplotlib
# ---------------------------------------------------------------------------

def visualize_graph(network: RoadNetwork, highlight_path: Optional[List[str]] = None):
    """Requires: pip install networkx matplotlib"""
    import networkx as nx
    import matplotlib.pyplot as plt

    G = nx.DiGraph()
    edge_colors = []
    for node, roads in network.graph.items():
        for road in roads:
            G.add_edge(node, road.to_node, congestion=road.congestion)

    pos = nx.spring_layout(G, seed=42)
    for u, v in G.edges():
        c = G[u][v]["congestion"]
        edge_colors.append((c, 0.6 * (1 - c), 0))  # green -> red as congestion rises

    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=1200,
            edge_color=edge_colors, width=2, arrows=True)

    if highlight_path and len(highlight_path) > 1:
        path_edges = list(zip(highlight_path, highlight_path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color="blue", width=4)

    plt.title("Road Network — edge color indicates congestion (green=clear, red=heavy)")
    plt.show()


# ---------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------

def build_sample_city() -> RoadNetwork:
    net = RoadNetwork()
    net.add_road("A", "B", distance_km=2.0, base_speed_kmh=50)
    net.add_road("B", "C", distance_km=3.5, base_speed_kmh=60)
    net.add_road("A", "D", distance_km=4.0, base_speed_kmh=50)
    net.add_road("D", "C", distance_km=2.5, base_speed_kmh=40)
    net.add_road("B", "D", distance_km=1.5, base_speed_kmh=35)
    net.add_road("C", "E", distance_km=3.0, base_speed_kmh=55)
    net.add_road("D", "E", distance_km=5.0, base_speed_kmh=60)
    return net


if __name__ == "__main__":
    random.seed(7)  # reproducible demo

    city = build_sample_city()
    nav = Navigator(city)

    # Run a few ticks to simulate live traffic before the user even asks
    for _ in range(3):
        nav.simulator.tick()

    # Simulate a real-world incident: accident jams the direct B->C route
    nav.simulator.simulate_incident("B", "C", severity=0.85)

    nav.get_directions("A", "E")

    # Uncomment to view the graph visually (requires networkx + matplotlib):
    # visualize_graph(city, highlight_path=nav.predictor.find_best_route("A", "E").path)
