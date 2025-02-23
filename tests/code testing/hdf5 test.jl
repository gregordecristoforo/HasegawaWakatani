using HDF5
# keys must be strings
# "group" is analogous to a directory
# "dataset" is like a file. HDF5 also uses "attributes" to associate metadata with a particular group or dataset
fid = h5open("test", "cw")

basename(tempname())*"-$(now())"
using Dates
now()

simulation_id = create_group(fid, "$(now())")
vels_id = create_group(simulation_id, "velocities")
data_id = create_group(simulation_id, "data")
whatever_id = create_group(simulation_id, "whatever")

# Creates dataset
vels_id["test"] = 0

read(vels_id["test"])
read(vels_id)

read(simulation_id["velocities"])

write(vels_id["myvector"], 2)

read(vels_id)

vels_id["myvector"][1] = [3]
vels_id["myvector"][1:4]


using OrderedCollections
p = OrderedDict(
    "" => 2,
    "a" => :n
)
vels_id["parameters"] = p

vels_id = fid["2025-02-21T16:56:50.900/velocities"]
#Chuncking
vels_id["A", chunk=(5,5)] = rand(100,10)
vels_id["A"] = rand(100,100)

# Should use chunking when writing entries of the vector

#close(fid)

i = 0
try
    while true      
        sleep(1e-10000000000000000)
    end
catch e
    println("Success")
    rethrow(e)
end

using OrderedCollections

A = OrderedDict(
    "hi" => 1,
    "3" => 2
)

for (key, val) in A 
    println(key)
end

A = rand(100,100)
g1["A", chunk=(5,5), compress=3] = A
g2["A", chunk=(5,5), shuffle=(), deflate=3] = A
using H5Zblosc # load in Blosc
g3["A", chunk=(5,5), blosc=3] = A

##vv Puts on the drive
dset = create_dataset(g, "B", datatype(Float64), dataspace(1000,100,10), chunk=(100,100,1))
dset[:,1,1] = rand(1000)
#^^ Incremently write to drive

initialize_diagnostics!
    #Create group
    create_group(diagnostic.name)

    #Dataset for time
    create_dataset("t", diagnostic.t)
    #Dataset for data
    create_dataset("data", diagnostic.data)
    #Attributes?
    diagnostic.label


a = fill(undef, (100, 100, 1000))
create_dataset(fid, "data2", a)
write(fid["data"], a)



# character

using HDF5
using Random
# Define dimensions
height, width = 100, 100  # Dimensions of your 2D arrays
# Create an HDF5 file
h5open("simulation_data.h5", "cw") do file
    # Create an extendable dataset
    dset = create_dataset(file, "timesteps10", datatype(Float32),
                          #(0, height, width),
                          (typemax(Int64), height, width),
                          chunk=(1, height, width),
                          compress=6)  # Compression level similar to gzip
    # Simulate appending data
    for i in 1:10  # Replace with your simulation loop
        # Generate a new 2D array (replace with your actual data)
        new_data = rand(Float32, height, width)
        # Extend the dataset along the time axis
        HDF5.set_extent_dims(dset, (i + 1, height, width))
        # Write the new data
        dset[i, :, :] = new_data
    end
end

fid = h5open("simulation_data.h5", "cw")
read(fid["timesteps10"])

typeof(fid)

g = create_group(fid, "simulation")
typeof(g)