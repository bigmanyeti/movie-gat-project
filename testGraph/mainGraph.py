import torch

x = torch.tensor([[20, 3],
                 [19, 4],
                 [18, 3]
                 ],dtype=torch.long)

edge_index = torch.tensor([[1,0,2],
                 [0,2,1]
                 ],dtype=torch.long)

edge_weight = torch.tensor([0.8,
                            0,5,
                            0.3],dtype=torch.long)

for i in range(edge_index.shape[1]):
    src = edge_index[0,i]
    dst = edge_index[1,i]

    print("source:", src.item())
    print("destination:",dst.item())

    print("src feat", x[src])
    print("dst feat", x[dst])
    print()



new_x = torch.zeros_like(x)
for i in range(edge_index.shape[1]):

    source = edge_index[0,i]
    des = edge_index[1,i]
    
    weight = edge_weight[i]

    mssg = weight * x[source]

    new_x[des] += mssg

print(new_x)