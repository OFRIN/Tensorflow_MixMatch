import numpy as np

def interleave_offsets(batch, nu):

    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1

    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
        
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [np.concatenate(v, axis=0) for v in xy]

# [0, 21, 42, 64]
print(interleave_offsets(64, 2))

mixed_input = [np.zeros((64, 32, 32, 3)) for i in range(3)]
mixed_input = interleave(mixed_input, 64)

print(np.shape(mixed_input))

