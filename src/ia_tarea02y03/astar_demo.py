import heapq
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def manhattan_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def create_grid(rows: int = 12, cols: int = 12) -> np.ndarray:
    grid = np.zeros((rows, cols), dtype=int)
    walls = [
        (2, 3), (2, 4), (2, 5), (2, 6),
        (3, 6), (4, 6), (5, 6), (6, 6), (7, 6),
        (8, 3), (8, 4), (8, 5), (8, 6),
    ]
    for r, c in walls:
        grid[r, c] = 1
    return grid


def get_neighbors(node: tuple[int, int], grid: np.ndarray) -> list[tuple[int, int]]:
    rows, cols = grid.shape
    r, c = node
    candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return [
        (nr, nc)
        for nr, nc in candidates
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0
    ]


def reconstruct_path(
    came_from: dict[tuple[int, int], tuple[int, int]],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    path = [goal]
    current = goal
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def run_search(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    use_heuristic: bool = True,
) -> dict:
    # priority queue entries store (f_score, h_score, insertion_id, node)
    h_start = manhattan_distance(start, goal) if use_heuristic else 0
    open_heap: list[tuple[int, int, int, tuple[int, int]]] = [(h_start, h_start, 0, start)]
    g_score: dict[tuple[int, int], int] = {start: 0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    closed_set: set[tuple[int, int]] = set()
    expansion_order: dict[tuple[int, int], int] = {}

    snapshots: list[dict] = []
    trace_log: list[dict] = []
    counter = 1

    while open_heap:
        f_curr, h_curr, _, current = heapq.heappop(open_heap)
        if current in closed_set:
            continue

        closed_set.add(current)
        expansion_order[current] = len(closed_set)
        pushed_neighbors: list[tuple[tuple[int, int], int, int, int]] = []

        # Expand valid orthogonal neighbors unless the goal has been popped
        if current != goal:
            for neighbor in get_neighbors(current, grid):
                if neighbor in closed_set:
                    continue
                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h_val = manhattan_distance(neighbor, goal) if use_heuristic else 0
                    f_val = tentative_g + h_val
                    heapq.heappush(open_heap, (f_val, h_val, counter, neighbor))
                    counter += 1
                    pushed_neighbors.append((neighbor, tentative_g, h_val, f_val))

        open_nodes = {item[3] for item in open_heap if item[3] not in closed_set}
        found_path = reconstruct_path(came_from, goal) if current == goal else []

        # Record frame state for step-by-step animation playback
        snapshots.append(
            {
                "current": current,
                "closed": set(closed_set),
                "open": open_nodes,
                "order": dict(expansion_order),
                "path": found_path,
            }
        )
        trace_log.append(
            {
                "step": len(closed_set),
                "node": current,
                "g": g_score[current],
                "h": h_curr,
                "f": f_curr,
                "pushed": pushed_neighbors,
            }
        )

        if current == goal:
            break

    final_path = snapshots[-1]["path"] if snapshots else []
    return {
        "path": final_path,
        "cost": len(final_path) - 1 if final_path else -1,
        "expanded": len(closed_set),
        "snapshots": snapshots,
        "trace": trace_log,
    }


def print_trace(trace_log: list[dict], max_entries: int = 12) -> None:
    print("=== A* Internal Search Trace (First 12 Expansions + Detour Point) ===")
    print(f"{'Step':<6}{'Popped':<12}{'g(n)':<7}{'h(n)':<7}{'f(n)':<7}{'Frontier Updates (node: g+h=f)'}")
    print("-" * 82)

    # Show initial greedy advance and the exact steps where A* escapes the concave wall
    indices = list(range(min(max_entries, len(trace_log))))
    for idx, entry in enumerate(trace_log):
        if idx >= max_entries and entry["f"] > 18 and len(indices) < max_entries + 4:
            indices.append(idx)
        elif idx >= max_entries and entry["f"] == 18 and len(indices) < max_entries + 6:
            indices.append(idx)

    for idx in sorted(set(indices)):
        entry = trace_log[idx]
        updates = ", ".join(
            f"{n}: {g}+{h}={f}" for n, g, h, f in entry["pushed"]
        ) or "dead-end / no improvement"
        print(
            f"{entry['step']:<6}{str(entry['node']):<12}"
            f"{entry['g']:<7}{entry['h']:<7}{entry['f']:<7}{updates}"
        )
    print("-" * 82 + "\n")


def print_comparison_summary(astar_res: dict, uninformed_res: dict) -> None:
    reduction = (1.0 - astar_res["expanded"] / uninformed_res["expanded"]) * 100.0
    print("=== Final Performance Comparison (12x12 Grid) ===")
    print(f"{'Algorithm':<30}{'Path Cost':<14}{'Nodes Expanded':<18}{'Reduction'}")
    print("-" * 76)
    print(
        f"{'A* (Manhattan Heuristic)':<30}{astar_res['cost']:<14}"
        f"{astar_res['expanded']:<18}{reduction:.1f}% fewer nodes"
    )
    print(
        f"{'Uninformed Search (h = 0)':<30}{uninformed_res['cost']:<14}"
        f"{uninformed_res['expanded']:<18}{'Baseline (0.0%)'}"
    )
    print("-" * 76)
    print(f"Optimal Path Found: {astar_res['path']}\n")


def animate_comparison(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    astar_res: dict,
    uninformed_res: dict,
    out_dir: Path,
) -> None:
    rows, cols = grid.shape
    cmap = ListedColormap([
        "#F8FAFC",  # 0: free cell
        "#1E293B",  # 1: obstacle
        "#FDE047",  # 2: open frontier
        "#93C5FD",  # 3: closed / explored
        "#FB923C",  # 4: current node
        "#34D399",  # 5: optimal path
        "#2563EB",  # 6: start
        "#EF4444",  # 7: goal
    ])
    norm = BoundaryNorm(np.arange(-0.5, 8.5, 1.0), cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4))
    titles = [
        r"A* Search ($f(n) = g(n) + h(n)$, Manhattan)",
        r"Uninformed Search ($h(n) = 0$, Dijkstra / BFS)",
    ]
    results = [astar_res, uninformed_res]

    images = []
    text_grids = []

    # Initialize static grid lines, cell labels, and coordinate axes for both panels
    for ax, title in zip(axes, titles):
        im = ax.imshow(grid, cmap=cmap, norm=norm, origin="upper")
        images.append(im)
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=10)
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which="minor", color="#CBD5E1", linestyle="-", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)

        cell_texts = []
        for r in range(rows):
            row_texts = []
            for c in range(cols):
                txt = ax.text(
                    c, r, "", ha="center", va="center",
                    fontsize=7.5, fontweight="bold", color="#0F172A"
                )
                row_texts.append(txt)
            cell_texts.append(row_texts)
        text_grids.append(cell_texts)

    legend_handles = [
        mpatches.Patch(color="#2563EB", label="Start (S)"),
        mpatches.Patch(color="#EF4444", label="Goal (G)"),
        mpatches.Patch(color="#1E293B", label="Obstacle"),
        mpatches.Patch(color="#FDE047", label="Frontier (Open)"),
        mpatches.Patch(color="#93C5FD", label="Explored (Closed)"),
        mpatches.Patch(color="#FB923C", label="Current"),
        mpatches.Patch(color="#34D399", label="Optimal Path"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=7,
        frameon=True,
        fontsize=8.5,
        bbox_to_anchor=(0.5, 0.01),
    )
    status_text = fig.suptitle("", fontsize=12.5, fontweight="bold", y=0.97)
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])

    max_steps = max(len(astar_res["snapshots"]), len(uninformed_res["snapshots"]))
    pause_frames = 18
    total_frames = max_steps + pause_frames

    # Update both grid panels at each animation frame and freeze on the final state
    def update(frame_idx: int):
        step_idx = min(frame_idx, max_steps - 1)
        for panel_idx, res in enumerate(results):
            snap = res["snapshots"][min(step_idx, len(res["snapshots"]) - 1)]
            board = grid.copy()

            for r, c in snap["open"]:
                board[r, c] = 2
            for r, c in snap["closed"]:
                board[r, c] = 3
            for r, c in snap["path"]:
                board[r, c] = 5

            curr_r, curr_c = snap["current"]
            if not snap["path"]:
                board[curr_r, curr_c] = 4

            board[start[0], start[1]] = 6
            board[goal[0], goal[1]] = 7
            images[panel_idx].set_data(board)

            for r in range(rows):
                for c in range(cols):
                    if (r, c) == start:
                        text_grids[panel_idx][r][c].set_text("S")
                        text_grids[panel_idx][r][c].set_color("white")
                    elif (r, c) == goal:
                        text_grids[panel_idx][r][c].set_text("G")
                        text_grids[panel_idx][r][c].set_color("white")
                    elif (r, c) in snap["order"]:
                        text_grids[panel_idx][r][c].set_text(str(snap["order"][(r, c)]))
                        text_grids[panel_idx][r][c].set_color("#0F172A")
                    else:
                        text_grids[panel_idx][r][c].set_text("")

        astar_step = min(step_idx + 1, astar_res["expanded"])
        uninf_step = min(step_idx + 1, uninformed_res["expanded"])
        status_text.set_text(
            f"Expansion Step {step_idx + 1}/{max_steps}  |  "
            f"A* Explored: {astar_step}/{astar_res['expanded']}  |  "
            f"Uninformed Explored: {uninf_step}/{uninformed_res['expanded']}"
        )
        return images

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=100,
        blit=False,
        repeat=True,
    )

    gif_path = out_dir / "astar_comparison.gif"
    png_path = out_dir / "astar_final.png"

    anim.save(gif_path, writer=animation.PillowWriter(fps=10))
    update(total_frames - 1)
    fig.savefig(png_path, dpi=300)

    print(f"Saved looping animation to: '{gif_path}'")
    print(f"Saved final comparison image to: '{png_path}'")
    plt.show()


def main() -> None:
    out_dir = PROJECT_ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = create_grid(rows=12, cols=12)
    start = (1, 1)
    goal = (10, 10)

    astar_res = run_search(grid, start, goal, use_heuristic=True)
    uninformed_res = run_search(grid, start, goal, use_heuristic=False)

    print_trace(astar_res["trace"])
    print_comparison_summary(astar_res, uninformed_res)
    animate_comparison(grid, start, goal, astar_res, uninformed_res, out_dir)


if __name__ == "__main__":
    main()
