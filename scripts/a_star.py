import heapq


def a_star(graph, start, goal):
    pq = [(0, start, [start], 0)]  # (f, node, path, g)
    visited = set()
    cost_dict = {start: 0}  

    while pq:
        _, node, path, g = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)

        if node == goal:
            return path, g  

        for neighbor in graph.edges[node]:
            new_g = g + graph.edges[node][neighbor]
            if neighbor not in cost_dict or new_g < cost_dict[neighbor]:
                cost_dict[neighbor] = new_g
                f = new_g + graph.heuristic(neighbor, goal)
                heapq.heappush(pq, (f, neighbor, path + [neighbor], new_g))

    return None, float('inf')