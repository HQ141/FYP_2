import ipfsApi
api = ipfsApi.Client('127.0.0.1', 5001)
res=api.add("simulaters/model_weights.hdf5")
print(res)