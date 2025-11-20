=== Graph Structure ===
HeteroData(
  user={ num_nodes=1000000 },
  tweet={ num_nodes=88217457 },
  hashtag={ num_nodes=5146289 },
  list={ num_nodes=21870 },
  (user, followers, user)={ edge_index=[2, 1116655] },
  (user, following, user)={ edge_index=[2, 2626979] },
  (list, followed, user)={ edge_index=[2, 493556] },
  (list, membership, user)={ edge_index=[2, 1022587] },
  (user, own, list)={ edge_index=[2, 21870] },
  (user, pinned, tweet)={ edge_index=[2, 347131] },
  (user, post, tweet)={ edge_index=[2, 88217457] },
  (list, contain, tweet)={ edge_index=[2, 1998788] },
  (tweet, discuss, hashtag)={ edge_index=[2, 66000633] },
  (tweet, mentioned, user)={ edge_index=[2, 4759388] },
  (user, like, tweet)={ edge_index=[2, 595794] },
  (tweet, replied_to, tweet)={ edge_index=[2, 1114980] },
  (tweet, retweeted, tweet)={ edge_index=[2, 1580643] },
  (tweet, quoted, tweet)={ edge_index=[2, 289476] }
)

=== Node Types & Counts ===
  - user: 1,000,000
  - tweet: 88,217,457
  - hashtag: 5,146,289
  - list: 21,870
  Total Nodes: 94,385,616

=== Edge Types & Counts ===
  - user --[followers]--> user: 1,116,655
  - user --[following]--> user: 2,626,979
  - list --[followed]--> user: 493,556
  - list --[membership]--> user: 1,022,587
  - user --[own]--> list: 21,870
  - user --[pinned]--> tweet: 347,131
  - user --[post]--> tweet: 88,217,457
  - list --[contain]--> tweet: 1,998,788
  - tweet --[discuss]--> hashtag: 66,000,633
  - tweet --[mentioned]--> user: 4,759,388
  - user --[like]--> tweet: 595,794
  - tweet --[replied_to]--> tweet: 1,114,980
  - tweet --[retweeted]--> tweet: 1,580,643
  - tweet --[quoted]--> tweet: 289,476
  Total Edges: 170,185,937

=== Features (User Node) ===
  - Found 'profile_features.pt':
    * Shape: torch.Size([1000000, 23])
    * Mean (first 5 dims): tensor([-0.1883, -0.1871, -0.1931, -0.1670, -0.1091])
    * Std  (first 5 dims): tensor([1.0061, 1.1019, 1.0188, 0.9780, 1.0376])
  - Found 'user_text_tokens.pt':
    * Input IDs Shape: torch.Size([1000000, 256])
    * Max Token ID: 249999

=== Connectivity Sanity Check ===
  - Users involved in 'followers' edges: 433,645 / 1,000,000

=== Labels ===
  - Total Labeled Users: 1000000 / 1000000
  - Human (0): 860,057
  - Bot (1):   139,943
  - Imbalance Ratio: 1 Bot : 6.1 Humans