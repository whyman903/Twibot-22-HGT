import torch
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
# from torch_geometric.utils import to_networkx
import sys
import os

# Add src to path so we can import modules if needed, 
# though we might just load the .pt file directly for inspection
sys.path.append(os.getcwd())

def inspect_graph(processed_dir="processed"):
    processed_path = Path(processed_dir)
    graph_path = processed_path / "hetero_graph.pt"
    
    if not graph_path.exists():
        print(f"Error: Graph file not found at {graph_path}")
        print("Have you run the training/preprocessing script yet?")
        return

    print(f"Loading graph from {graph_path}...")
    try:
        # Use weights_only=False for compatibility as per your codebase's pattern
        try:
            data = torch.load(graph_path, weights_only=False)
        except TypeError:
            data = torch.load(graph_path)
            
        print("\n=== Graph Structure ===")
        print(data)
        
        print("\n=== Node Types & Counts ===")
        total_nodes = 0
        for node_type in data.node_types:
            count = data[node_type].num_nodes
            total_nodes += count
            print(f"  - {node_type}: {count:,}")
        print(f"  Total Nodes: {total_nodes:,}")

        print("\n=== Edge Types & Counts ===")
        total_edges = 0
        for edge_type in data.edge_types:
            count = data[edge_type].edge_index.shape[1]
            total_edges += count
            src, rel, dst = edge_type
            print(f"  - {src} --[{rel}]--> {dst}: {count:,}")
        print(f"  Total Edges: {total_edges:,}")

        print("\n=== Features (User Node) ===")
        # Check if user features are attached directly or stored separately
        if 'x' in data['user']:
            print(f"  - User feature shape: {data['user'].x.shape}")
        
        # Check Profile Features
        profile_path = processed_path / "profile_features.pt"
        if profile_path.exists():
            print("  - Found 'profile_features.pt':")
            try:
                try:
                    profile_feat = torch.load(profile_path, weights_only=False)
                except TypeError:
                    profile_feat = torch.load(profile_path)
                print(f"    * Shape: {profile_feat.shape}")
                print(f"    * Mean (first 5 dims): {profile_feat.float().mean(dim=0)[:5]}")
                print(f"    * Std  (first 5 dims): {profile_feat.float().std(dim=0)[:5]}")
            except Exception as e:
                print(f"    * Error loading profile features: {e}")
        else:
            print("  - 'profile_features.pt' not found.")

        # Check Text Tokens
        text_path = processed_path / "user_text_tokens.pt"
        if text_path.exists():
            print("  - Found 'user_text_tokens.pt':")
            try:
                try:
                    text_tokens = torch.load(text_path, weights_only=False)
                except TypeError:
                    text_tokens = torch.load(text_path)
                if isinstance(text_tokens, dict) and 'input_ids' in text_tokens:
                    print(f"    * Input IDs Shape: {text_tokens['input_ids'].shape}")
                    print(f"    * Max Token ID: {text_tokens['input_ids'].max()}")
                else:
                    print(f"    * Unexpected format: {type(text_tokens)}")
            except Exception as e:
                print(f"    * Error loading text tokens: {e}")
        else:
            print("  - 'user_text_tokens.pt' not found.")

        # Basic Connectivity Check
        print("\n=== Connectivity Sanity Check ===")
        # Check for isolated nodes (simplified check for users)
        if ('user', 'followers', 'user') in data.edge_types:
            edge_index = data[('user', 'followers', 'user')].edge_index
            active_users = torch.unique(edge_index.flatten())
            print(f"  - Users involved in 'followers' edges: {len(active_users):,} / {data['user'].num_nodes:,}")
        
        # Check Labels
        label_path = processed_path / "user_labels.pt"
        if label_path.exists():
             print("\n=== Labels ===")
             try:
                try:
                    labels = torch.load(label_path, weights_only=False)
                except TypeError:
                    labels = torch.load(label_path)
                
                valid_labels = labels[labels >= 0]
                print(f"  - Total Labeled Users: {valid_labels.numel()} / {labels.numel()}")
                
                humans = (valid_labels == 0).sum().item()
                bots = (valid_labels == 1).sum().item()
                print(f"  - Human (0): {humans:,}")
                print(f"  - Bot (1):   {bots:,}")
                if bots > 0:
                    print(f"  - Imbalance Ratio: 1 Bot : {humans/bots:.1f} Humans")
             except Exception as e:
                 print(f"  - Error loading labels: {e}")

        # visualize_subgraph(data)
        visualize_subgraph(data)

    except Exception as e:
        print(f"Error inspecting graph: {e}")
        import traceback
        traceback.print_exc()

def visualize_subgraph(data):
    print("\n=== Visualizing Sample Subgraph ===")
    try:
        G = nx.DiGraph()
        
        # Find a user with many connections (hub) to make the graph interesting
        # We'll look at the 'following' relation to find a popular user
        print("  Searching for a well-connected user...")
        if ('user', 'following', 'user') in data.edge_types:
            edge_index = data[('user', 'following', 'user')].edge_index
            # Count occurrences of source nodes (users who follow many people)
            # or target nodes (users who are followed by many)
            # Let's pick a user who follows at least 5 people
            unique_src, counts = torch.unique(edge_index[0], return_counts=True)
            candidates = unique_src[counts >= 5]
            if len(candidates) > 0:
                center_node_idx = candidates[0].item()
            else:
                center_node_idx = 0
        else:
            center_node_idx = 0
            
        print(f"  Selected Center User: {center_node_idx}")
        
        # Add center node
        G.add_node(f"u{center_node_idx}", type="user", color="#1f77b4", label="Center User")
        
        relations_to_plot = [
            ('user', 'followers', 'user'),
            ('user', 'following', 'user'),
            # ('user', 'post', 'tweet') # Skipping tweets to focus on social graph per user request
        ]
        
        max_nodes_per_type = 10
        
        for etype in relations_to_plot:
            if etype not in data.edge_types:
                continue
                
            src_type, rel, dst_type = etype
            edge_index = data[etype].edge_index
            
            # Edges where center is source (e.g. User -> follows -> Other)
            if src_type == 'user':
                mask = edge_index[0] == center_node_idx
                targets = edge_index[1][mask].tolist()[:max_nodes_per_type]
                for t in targets:
                    node_id = f"u{t}"
                    if node_id not in G:
                        G.add_node(node_id, type="user", color="#aec7e8") # Lighter blue for neighbors
                    G.add_edge(f"u{center_node_idx}", node_id, relation=rel)
            
            # Edges where center is target (e.g. Other -> follows -> User)
            if dst_type == 'user':
                mask = edge_index[1] == center_node_idx
                sources = edge_index[0][mask].tolist()[:max_nodes_per_type]
                for s in sources:
                    node_id = f"u{s}"
                    if node_id not in G:
                        G.add_node(node_id, type="user", color="#aec7e8")
                    G.add_edge(node_id, f"u{center_node_idx}", relation=rel)
        
        if G.number_of_nodes() > 1:
            plt.figure(figsize=(12, 12))
            
            # Use a layout that emphasizes the center
            pos = nx.spring_layout(G, k=0.5, iterations=50)
            
            colors = [nx.get_node_attributes(G, 'color').get(n, 'gray') for n in G.nodes()]
            
            nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=800, alpha=0.9)
            nx.draw_networkx_labels(G, pos, font_size=8)
            
            # Draw edges with different styles for different relations
            edge_labels = nx.get_edge_attributes(G, 'relation')
            
            # Separate edges by type for potentially different styling (optional, keeping simple for now)
            nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20, edge_color='gray', alpha=0.5)
            
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, label_pos=0.3)
            
            plt.title(f"Social Subgraph for User {center_node_idx}", fontsize=16)
            plt.axis('off')
            
            out_path = "graph_sample_viz.png"
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"  Saved visualization to {out_path}")
        else:
            print(f"  User {center_node_idx} has no social connections in the sampled graph.")

    except Exception as e:
        print(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_graph()
